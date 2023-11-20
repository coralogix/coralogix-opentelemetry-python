# coralogix-opentelemetry-python
coralogix opentelemetry contribution for python

## Development
This project uses [poetry](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions) for dependency management.
you can run the `install` script to install all poetry and all dependencies.

we use flake8 for linting, black for formatting and mypy for static type checking
ro run full linting and type checking run the `lint` script

### Installation

### Scripts
all scripts are found under `scripts`
* `install.sh` - install poetry and all dependencies
* `build.sh` - install poetry and all non dev dependencies
* `lint.sh` - run linting and type checking
* `test.sh` - and lint.sh and unit tests

### Intellij
It is recommend to configure intellij to work with formatting of black and type checking of mypy
for mypy install the mypy plugin and configure it to use the `mypy.ini` file
for black install the BlackConnect plugin, configure the local blackd server connect to it
make sure to check The `Trigger on code reformat` checkbox