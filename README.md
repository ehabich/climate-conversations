# Climate Conversations
This repository uses climate change scientific papers and Reddit conversations to summarize research and public concerns around climate change for policymakers.

### Project Requirements
- Python version: `^3.11`
- [Poetry](https://python-poetry.org/)

### Instructions to Run the Project
1. Go into the base directory of the repository and type `poetry shell` into the terminal.
2. Use the `make run` command.

### Technical Notes
- Any modules should be added via the `poetry add [module]` command.
  - Example: `poetry add black`

## Standard Commands
- `make lint`: Runs `pre-commit`.
- `make test`: Runs test cases in the `test` directory.
- `make run`: Runs the `main` function in the `project` folder.
- `make`: Runs `lint` and adds poetry requirements to `requirements.txt`
