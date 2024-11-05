import os
import sys
import io
import click
import polib
import tomllib
import json
from subprocess import check_output, CalledProcessError
from datetime import datetime
from openai import OpenAI
from pprint import pprint
from instructor import from_openai, Mode, Instructor
from pydantic import BaseModel, Field, RootModel
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from dotenv import load_dotenv
from typing import Generator, Iterable, Iterator, List, Optional, Tuple, Dict, cast
from concurrent.futures import ThreadPoolExecutor


def read_pot_file(filename: str) -> str:
    with io.open(
        filename, "r", encoding="utf-8", buffering=io.DEFAULT_BUFFER_SIZE
    ) as file:
        return file.read()


def write_po_file(filename: str, content: str) -> None:
    with io.open(
        filename, "w", encoding="utf-8", buffering=io.DEFAULT_BUFFER_SIZE
    ) as file:
        file.write(content)


def log_instructor_kwargs(**kwargs):
    pprint(f"Function called with kwargs: {kwargs}")


def log_instructor_exception(exception: Exception):
    pprint(f"An exception occurred: {str(exception)}")


def get_language_code(client: Instructor, model: str, language: str) -> str:
    class LanguageCode(BaseModel):
        data: str = Field(description="The two letter ISO 639-1 language code")

    language_code = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a ISO 639-1 language code generator. You receive a language name and return the corresponding two character language code.",
            },
            {"role": "user", "content": f"{language.lower()}"},
        ],
        response_model=LanguageCode,
    )

    return language_code.data


def translate(client: Instructor, model: str, pot_content: str, language: str):
    class Translation(BaseModel):
        """
        Represents a single translation entry in a Portable Object (PO) file.

        Each translation entry contains an original message ID (`msgid`) and a
        translated string (`msgstr`).

        Attributes:
            msgid (str): The original source language that requires translation.
            msgstr (str): The translated text in the target language.
        """

        msgid: str = Field(..., description="The msgid ID")
        msgstr: str = Field(..., description="The translated msgstr")

        class Config:
            extra = 'forbid'

    class Translations(BaseModel):
        """
        A collection of translation entries for a Portable Object (PO) file.

        The model holds a list of `Translation` objects, each containing an `msgid`
        and `msgstr`.

        Attributes:
            translations (List[Translation]): List of translation entries, where each entry
                is an object with the original string ID (`msgid`) and its Spanish translation (`msgstr`).
                - Example: [{"msgid": "<original_string>", "msgstr": "<translated_string>"}]
        """
        data: List[Translation] = Field(
            ..., description="List of translations where each item contains an 'msgid' and 'msgstr'."
        )

        class Config:
            extra = 'forbid'

    translations = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are a perfect PO (Portable Object) translation system. You receive POT (Portable Object Template) file content, with strings using ICU MessageFormat patterns. You return msgid's with their translated msgstr's in {language.title()}."},
            {"role": "user", "content": pot_content},
        ],
        response_model=Translations,
    )

    # raw_json = translations_completion.choices[0].message.tool_calls[0].function.arguments
    # parsed_data = json.loads(raw_json)
    # translations_list = parsed_data["data"]
    # for translation in translations_list:
    #     print(f"Original Message ID: {translation['msgid']}")
    #     print(f"Translated Message: {translation['msgstr']}\n")

    return translations.data


def generate_po_file(client: Instructor, model: str, pot_content: str, language: str):
    po = polib.pofile(pot_content)

    po.header = ""
    po.metadata["PO-Revision-Date"] = datetime.now().strftime("%Y-%m-%d %H:%M%z")
    po.metadata["X-Generator"] = "A17R 0.1.0"
    try:
        name = check_output(["git", "config", "user.name"]).decode().strip()
        email = check_output(["git", "config", "user.email"]).decode().strip()
        translator_info = f"{name} <{email}>"
    except Exception as e:
        print("Could not obtain translator info from user's machine.")
        translator_info = "a17r <support@example.net>"
    po.metadata["Last-Translator"] = translator_info

    translations = translate(client, model, pot_content, language)

    translations_dict = {translation.msgid: translation.msgstr for translation in translations}

    for entry in po:
        if entry.msgid in translations_dict:
            entry.msgstr = translations_dict[entry.msgid]

    return po


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("pot_file_path", type=click.Path(exists=True))
@click.option(
    "--language",
    "-l",
    type=str,
    required=True,
    multiple=True,
    help="Target language for translation.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="gpt-3.5-turbo",
    help="Tool use capable language model for translation. Default is 'gpt-3.5-turbo'.",
)
@click.option("--api-key", "-k", type=str, help="Inference service API key.")
@click.option(
    "--base-url",
    "-b",
    type=str,
    help="Base url for a OpenAI API compatible inference service.",
)
def cli(
    pot_file_path: str,
    language: List[str],
    model: str,
    api_key: Optional[str],
    base_url: Optional[str] = None,
):
    load_dotenv(".env.local")
    load_dotenv(".env", override=False)

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        click.echo(
            "API key is required. Please provide one using --api-key or set OPENAI_API_KEY in a .env file.",
            err=True,
        )
        sys.exit(1)

    languages_list = list(language)

    openai_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
    }
    openai_kwargs = {k: v for k, v in openai_kwargs.items() if v is not None}
    client = from_openai(
        OpenAI(**openai_kwargs), mode=Mode.TOOLS
    )  # mode=Mode.TOOLS_STRICT

    # client.on("completion:kwargs", lambda *args, **kwargs: log_instructor_kwargs(**kwargs))
    client.on("completion:error", log_instructor_exception)

    pot_content = read_pot_file(pot_file_path)

    with ThreadPoolExecutor() as executor:
        futures = {}

        for lang in languages_list:
            future = executor.submit(
                generate_po_file,
                client,
                model,
                pot_content,
                lang
            )
            futures[future] = lang

        for future in futures:
            language_code = get_language_code(client, model, language=futures[future])
            po_filename = f"{language_code}.po"
            po = future.result()
            po.save(po_filename)
            click.echo(
                f"Translation to '{futures[future]}' complete. Output written to '{po_filename}'"
            )


if __name__ == "__main__":
    cli()
