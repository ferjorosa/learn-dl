1.  Create a poetry.toml file with content:

```
[virtualenvs]
in-project = true
```

2. `poetry init --python ">=3.10.6,<3.11"`
3. You can add sources like `poetry source add cloudsmith https://dl.cloudsmith.io/...`
4. Run the following to create and activate the environment:

```
poetry install
poetry shell
```
4. Add libraries

```
poetry add aily-py-commons@^0.1.18
poetry add aily-ai-brain@^1.0.0
```

5.  Add pre-commit

```
poetry add --group dev pre-commit@^3.5.0
```

6. Create `pre-commit-config.yaml` file

7. Run `pre-commit install`