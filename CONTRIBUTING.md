# Contributing to insta_rag

Thank you for your interest in contributing to `insta_rag`! This guide will help you get started with the development workflow, tools, and best practices.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Development Workflow](#development-workflow)
- [Code Quality Tools](#code-quality-tools)
- [Version Management](#version-management)
- [Commit Guidelines](#commit-guidelines)
- [Branch Naming Convention](#branch-naming-convention)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Prerequisites

Before you begin, ensure you have the following tools installed on your system:

### Required Tools

1. **Python 3.9 or higher**

   - **Windows**: Download from [python.org](https://www.python.org/downloads/)
   - **macOS**: Use Homebrew: `brew install python@3.9` or download from [python.org](https://www.python.org/downloads/)
   - **Linux**: Usually pre-installed. If not: `sudo apt install python3 python3-pip` (Debian/Ubuntu) or `sudo yum install python3 python3-pip` (RHEL/CentOS)

1. **Git**

   - **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
   - **macOS**: `brew install git` or comes with Xcode Command Line Tools: `xcode-select --install`
   - **Linux**: `sudo apt install git` (Debian/Ubuntu) or `sudo yum install git` (RHEL/CentOS)

1. **uv** (Python package manager)

   uv is a fast Python package installer and resolver, used for managing dependencies in this project.

   - **Windows**:

     ```powershell
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

   - **macOS/Linux**:

     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   After installation, verify it's working:

   ```bash
   uv --version
   ```

### Optional Tools (Recommended)

- **GPG** (for signing commits and tags) - Configured in commitizen settings
  - **Windows**: Install [Gpg4win](https://gpg4win.org/download.html)
  - **macOS**: `brew install gnupg`
  - **Linux**: Usually pre-installed. If not: `sudo apt install gnupg` (Debian/Ubuntu)

## Setting Up Your Development Environment

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/insta_rag.git
cd insta_rag
```

### 2. Install Development Dependencies

The project uses `uv` for dependency management. Install all dependencies including development tools:

```bash
# Install the package in editable mode with dev dependencies
uv pip install -e . --group dev
```

This will install:

- All runtime dependencies (listed in `pyproject.toml` under `dependencies`)
- Development tools: `commitizen`, `ruff`, `pre-commit`, `pytest`, etc. (listed under `[dependency-groups].dev`)

### 3. Set Up Pre-commit Hooks

Pre-commit hooks automatically check your code before each commit to maintain quality standards.

```bash
# Install pre-commit hooks
pre-commit install

# Install the commit message hook (for conventional commits)
pre-commit install --hook-type commit-msg

# Install the pre-push hook (for branch naming validation)
pre-commit install --hook-type pre-push
```

The pre-commit hooks will:

- Lock and export dependencies with `uv`
- Check for trailing whitespace
- Ensure consistent line endings (LF)
- Run `ruff` for linting and formatting
- Format Markdown files with `mdformat`
- Validate commit messages follow conventional commits format
- Validate branch names follow conventional branch naming

### 4. Configure Git (Optional but Recommended)

If you plan to use GPG signing (as configured in the project):

```bash
# Configure your GPG key
git config --global user.signingkey YOUR_GPG_KEY_ID
git config --global commit.gpgsign true
git config --global tag.gpgsign true
```

**Note**: If you don't have GPG configured, you may need to temporarily disable signing when using commitizen. You can do this by setting `gpg_sign = false` in your local `.cz.toml` file, but do not commit this change.

## Development Workflow

### Making Changes

1. **Create a new branch** following the [branch naming convention](#branch-naming-convention):

   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

1. **Make your changes** to the codebase

1. **Run linters and formatters**:

   ```bash
   # Check for linting errors and auto-fix them
   ruff check . --fix

   # Format the codebase
   ruff format .
   ```

1. **Run tests** (if applicable):

   ```bash
   pytest
   ```

1. **Commit your changes** following [conventional commits](#commit-guidelines):

   ```bash
   # Using commitizen (recommended)
   cz commit

   # Or manually with a conventional commit message
   git commit -m "feat: add new feature"
   ```

1. **Push your branch**:

   ```bash
   git push origin feat/your-feature-name
   ```

## Code Quality Tools

This project uses several tools to maintain code quality:

### Ruff

Ruff is used for both linting and formatting Python code.

**Configuration**: Defined in `pyproject.toml` under `[tool.ruff]`

- Line length: 88 characters (Black-compatible)
- Target: Python 3.9+
- Double quotes for strings
- LF line endings

**Usage**:

```bash
# Lint and auto-fix issues
ruff check . --fix

# Format code
ruff format .

# Check without fixing
ruff check .
```

### Pre-commit

Pre-commit runs automated checks before commits. It includes:

- **uv-lock**: Ensures `uv.lock` is up-to-date
- **uv-export**: Exports dependencies to `requirements.txt`
- **trailing-whitespace**: Removes trailing whitespace
- **mixed-line-ending**: Ensures LF line endings
- **ruff-check**: Lints Python code
- **ruff-format**: Formats Python code
- **mdformat**: Formats Markdown files
- **commitizen**: Validates commit messages
- **commitizen-branch**: Validates branch names

**Usage**:

```bash
# Run all hooks manually
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff-check --all-files
```

### Commitizen

Commitizen enforces conventional commits and automates version bumping.

**Configuration**: Defined in `pyproject.toml` under `[tool.commitizen]`

## Version Management

The project uses semantic versioning with pre-release tags (e.g., `0.1.0-beta.2`). Version information is stored in three locations:

1. `pyproject.toml` (`version` field)
1. `uv.lock` (automatically synced)
1. Git tags (e.g., `v0.1.0-beta.2`)

### Option 1: Using Commitizen (Recommended)

Commitizen automates the version bumping process:

```bash
# Bump a pre-release version (for beta releases)
cz bump --prerelease beta

# This will:
# 1. Update the version in pyproject.toml
# 2. Run pre-bump hooks (uv lock, git add uv.lock)
# 3. Update CHANGELOG.md
# 4. Create a signed, annotated Git tag (v0.1.0-beta.X)
# 5. Commit all changes
```

**Other version bump commands**:

```bash
# Bump patch version (0.1.0 -> 0.1.1)
cz bump --patch

# Bump minor version (0.1.0 -> 0.2.0)
cz bump --minor

# Bump major version (0.1.0 -> 1.0.0)
cz bump --major

# Bump pre-release (0.1.0-beta.1 -> 0.1.0-beta.2)
cz bump --prerelease beta
```

**Important**: After running `cz bump`, push both commits and tags:

```bash
git push origin your-branch
git push origin --tags
```

### Option 2: Manual Version Update

If you need to update the version manually (not recommended unless necessary):

1. **Update `pyproject.toml`**:

   ```toml
   [project]
   version = "0.1.0-beta.3"  # Update this line
   ```

1. **Lock dependencies** to sync `uv.lock`:

   ```bash
   uv lock
   ```

1. **Create a Git tag**:

   ```bash
   # Create an annotated, signed tag
   git tag -a -s v0.1.0-beta.3 -m "Release v0.1.0-beta.3"
   ```

1. **Commit and push**:

   ```bash
   git add pyproject.toml uv.lock
   git commit -m "chore: bump version to 0.1.0-beta.3"
   git push origin your-branch
   git push origin v0.1.0-beta.3
   ```

**Warning**: Manual version updates are error-prone. Always ensure:

- The version in `pyproject.toml` matches the Git tag (minus the `v` prefix)
- The `uv.lock` file is updated by running `uv lock`
- All three locations are synchronized

## Commit Guidelines

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Examples

```bash
# Feature
feat: add semantic chunking support

# Bug fix
fix: resolve memory leak in document processing

# Documentation
docs: update installation instructions

# Refactoring
refactor: simplify retrieval pipeline

# Multiple scopes
feat(api): add new endpoint for batch processing
```

### Using Commitizen for Commits

The easiest way to create valid commit messages is using the interactive prompt:

```bash
cz commit
```

This will guide you through:

1. Selecting the commit type
1. Entering the scope (optional)
1. Writing a short description
1. Adding a longer description (optional)
1. Noting breaking changes (optional)
1. Referencing issues (optional)

## Branch Naming Convention

This project follows the [Conventional Branch](https://conventional-branch.github.io) naming convention, which is validated by pre-commit hooks.

### Branch Name Format

```
<type>/<description>
```

### Branch Types

Use the same types as commits:

- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications
- `chore/` - Maintenance tasks
- `ci/` - CI/CD changes

### Examples

```bash
# Good branch names
git checkout -b feat/add-hybrid-retrieval
git checkout -b fix/memory-leak-in-chunker
git checkout -b docs/update-contributing-guide
git checkout -b refactor/simplify-embedder-interface

# Bad branch names (will be rejected by pre-push hook)
git checkout -b new-feature          # Missing type prefix
git checkout -b feature/add-stuff    # Wrong type (should be 'feat')
git checkout -b fix/Fix-Bug          # Inconsistent casing
```

## Testing

The project uses `pytest` for testing.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/insta_rag --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run tests with verbose output
pytest -v
```

### Test Configuration

Test configuration is defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

- Test paths: `tests/` and `examples/`

### Writing Tests

- Place unit tests in the `tests/` directory
- Name test files with the `test_` prefix
- Name test functions with the `test_` prefix
- Use descriptive test names that explain what is being tested

## Pull Request Process

1. **Ensure all checks pass**:

   - Pre-commit hooks run successfully
   - All tests pass
   - Code is properly formatted and linted

1. **Update documentation** if you've made changes that affect:

   - Public APIs
   - Configuration options
   - Installation or setup procedures

1. **Write a clear PR description**:

   - Explain what changes you made and why
   - Reference any related issues (e.g., "Fixes #123")
   - Include screenshots if you've made UI changes (if applicable)

1. **Follow conventional commits** in your PR:

   - PR titles should follow the conventional commit format
   - Squash commits if you have many small commits

1. **Request a review**:

   - Tag relevant maintainers or contributors
   - Respond to feedback promptly
   - Make requested changes in new commits

1. **Wait for approval**:

   - At least one maintainer approval is typically required
   - CI checks must pass
   - Any merge conflicts must be resolved

## Getting Help

If you have questions or need help:

- **Issues**: Check existing [GitHub Issues](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/issues)
- **Discussions**: Start a discussion in [GitHub Discussions](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/discussions)
- **Documentation**: Refer to the [project documentation](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki)

## Code of Conduct

Please note that this project follows standard open-source etiquette. Be respectful, constructive, and collaborative in all interactions.

______________________________________________________________________

**Thank you for contributing to insta_rag!** Your contributions help make this project better for everyone.
