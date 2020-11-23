# Contributor guidelines

## General information

If you wish to contribute, make sure you've discussed it with me in an issue.

Your words and actions should at all times be in line with the [code of conduct](CODE_OF_CONDUCT.md).

## Code conventions

You can set up all development dependencies with the `requirements-dev.txt` file. Please use Python 3.6 or higher.

- Format the code with `black --line-length 99`
- Lint the code with `flake8`
- Sort imports following the `appnexus` convention, and split (line-wise) multiple imports from the same package
- Commits should be signed using `git commit -s`

Please test your code before opening a PR, and place new features in separate files. If possible, do not add any dependencies.
