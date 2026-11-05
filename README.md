# NLP1 2026 – Groep 4

## Setup (eenmalig)

Je hebt [uv](https://docs.astral.sh/uv/getting-started/installation/) nodig.

```bash
git clone https://github.com/jeffkroon/NLP_groep_4.git
cd NLP_groep_4
uv sync                        # installeert Python 3.14 + exact dezelfde packages (uv.lock)
uv run nbstripout --install    # VERPLICHT: haalt notebook-output weg bij committen
```

Open de notebooks in VS Code en kies als kernel `.venv` (`./.venv/bin/python`), of draai `uv run jupyter lab`.

## Structuur

| Map | Wat |
|-----|-----|
| `part1/` | Tutorialcode van het vak – niet aanpassen |
| `src/` | Gedeelde code: `data` (laden + CV-splits), `features` (representaties), `models` (architecturen), `evaluate` (CV, macro-F1, Bayesiaanse vergelijking), `plots` (figuren) |
| `notebooks/` | Verkennen en plotten, één notebook per persoon (`naam_onderwerp.ipynb`) |
| `results/` | Kleine csv's met scores |
| `scripts/` | Bijv. het downloaden van de dataset |

Data en modelbestanden staan **niet** in git (zie `.gitignore`).

## Werkwijze

```bash
git pull                          # altijd eerst
git switch -c <taak>              # branch per taak, bv. mlp-dropout
# werken + committen
git push -u origin <taak>         # daarna Pull Request op GitHub, laat iemand reviewen
```

- Echte code in `src/`, notebooks roepen die aan. Nooit met z'n tweeën in hetzelfde notebook.
- Gebruik `SEEDS` en `N_FOLDS` uit `src/data.py`, zodat ieders resultaten vergelijkbaar zijn.
- Data: `load_stance_all()` laadt alle vijf stance-topics (met `[topic]`-tag voor elke tweet); `load_stance(topic)` één topic.
- Nieuwe representatie of model? Voeg hem toe aan `ENCODER_FACTORIES` (`src/features.py`) of `MODEL_RUNNERS` (`src/models.py`); de experimenten en figuren in het notebook pakken hem dan vanzelf mee.
- Hoofdnotebook met alle experimenten: `notebooks/jeff_tokenizer.ipynb`. Na een wijziging in `src/`: kernel herstarten.
- Packages toevoegen: `uv add <package>` en commit `pyproject.toml` + `uv.lock`.
- AI gebruikt? Zet het in [`AI-GEBRUIK.md`](AI-GEBRUIK.md).

---

# Originele instructies van het vak

We are using python 3.14, here is how you reproduce our setup.

```bash
python3.14 -m venv ~/envs/nlp1-2026
source ~/envs/nlp1-2026/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Each `part{N}` folder contains a tutorial notebook `tutorial.ipynb` and project instructions in `project.ipynb`.

## Alternatives

### Using `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast, modern Python package manager that is quickly becoming the industry standard. It is fully compliant with Python standards, fully replaces `pip`, and interoperates well with PyTorch. To use it for this project:

1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

2. Change the working directory:

    ```shell
    cd ./nlp1-student
    ```

3. Initialize the project

    ```shell
    uv init --bare .
    ```

    Notice the added file `pyproject.toml` file. This is where all metadata about this project/directory gets stored. You can read more about it [here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

    At the moment, it should look something like:

    ```toml
    [project]
    name = "nlp1-student"
    version = "0.1.0"
    requires-python = ">=3.14"
    dependencies = []
    ```

4. Install dependencies

    We're going to read the required dependencies from the provided `requirements.txt` file, and let `uv` handle installing the right versions

    ```shell
    uv add -r requirements.txt
    ```

    Note the new folder `.venv`. This is where all our dependencies live. Feel free to delete it once you're done, `uv` can quickly reconstruct it from its cache.

    In `pyproject.toml` the `dependencies` argument is now populated.

We can now use this virtual environment as we before

```shell
source .venv
```

Most modern IDEs (e.g., vscode or vscodium) should be able to recognize this virtual environment and load it for you.

### Using `vscode`

[Visual Studio Code (vscode)](https://code.visualstudio.com/) and its open-source twin [vscodium](https://vscodium.com/) are modern language agnostic text editors (read, IDE-lite). You can also use them for working with Jupyter or `.ipynb` notebooks. While these do not enjoy first-class support, the benefit of an IDE is that you can also access the terminal, edit regular `.py` files, and read all other files in the directory.

To use either for this project, we'll need to do a few extra steps.

1. Install and enable the [`Python`](https://open-vsx.org/vscode/item?itemName=ms-python.python) extension
2. Install and enable the [`Jupyter`](https://open-vsx.org/vscode/item?itemName=ms-toolsai.jupyter) extension
3. If necessary, install the `jupyter` package

    ```shell
    uv add jupyter 
    ```

4. When opening the notebook, use the 'Select kernel' dropdown, and select `.venv` (located as `./.venv/bin/python`)

You should now be able to read, edit and run the various `*.ipynb` notebooks in this directory

You can read more about the features vscode has for notebooks [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

#### Mermaid Diagrams

Some of the notebooks use [`mermaid`](https://mermaid.js.org/) diagrams. For example:

```mermaid
    flowchart LR

    A["foo"] --> B(bar) --> C["baz"]
```

If these are not rendering automatically (i.e., you see text instead of a diagram), you might need to upgrade your vscode instance to at least version [1.121.0](https://code.visualstudio.com/updates/v1_121#_mermaid-diagrams-in-markdown-preview-and-notebooks). If this is not possible, you should also be able to replicate this effect by installing the [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid).
