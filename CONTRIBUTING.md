# Contributing to Tinygrad

Thank you for your interest in contributing to Tinygrad! We welcome contributions from the community to help make this project better. This document outlines the process to contribute effectively and ensures a smooth collaboration.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Code of Conduct](#code-of-conduct)
3. [Ways to Contribute](#ways-to-contribute)
4. [Development Workflow](#development-workflow)
5. [Coding Guidelines](#coding-guidelines)
6. [Testing and Documentation](#testing-and-documentation)
7. [Submitting a Pull Request](#submitting-a-pull-request)
8. [Getting Help](#getting-help)

---

## Getting Started

1. **Fork the Repository:** Start by forking the Tinygrad repository to your GitHub account.
2. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/tinygrad.git
   ```
3. **Install Dependencies:** Refer to the `README.md` for instructions on setting up the environment.
4. **Explore the Codebase:** Familiarize yourself with the project structure and existing functionality.

---

## Code of Conduct

We adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please ensure your interactions are respectful and constructive.

---

## Ways to Contribute

- **Bug Reports:** Found a bug? Check the [Issues](https://github.com/tinygrad/tinygrad/issues) page to see if it's already reported. If not, create a new issue with details and reproduction steps.
- **Feature Requests:** Have an idea for a new feature? Open an issue and describe your proposal.
- **Code Contributions:** Tackle existing issues or implement new features.
- **Documentation Improvements:** Enhance the project documentation or fix errors.
- **Testing:** Write test cases or improve existing tests to increase code coverage.

---

## Development Workflow

1. **Create a Branch:**
   ```bash
   git checkout -b <feature-or-bugfix-branch>
   ```
2. **Make Changes:** Write clean, modular, and well-documented code.
3. **Run Tests:** Ensure all tests pass before committing your changes.
   ```bash
   pytest
   ```
4. **Commit Your Changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
5. **Push Your Changes:**
   ```bash
   git push origin <feature-or-bugfix-branch>
   ```

---

## Coding Guidelines

- **Follow PEP 8:** Adhere to Python's PEP 8 guidelines.
- **Write Clear Commit Messages:** Clearly explain the "why" and "what" of your changes.
- **Keep Changes Focused:** Limit each pull request to a single feature or fix.
- **Add Type Hints:** Use type annotations for better code readability and maintainability.
- **Document Your Code:** Include docstrings for functions, classes, and modules.

---

## Testing and Documentation

- **Tests:**
  - Add or update tests to cover your changes.
  - Use `pytest` for running the tests.
- **Documentation:**
  - Update the `README.md` or other documentation files if your changes impact usage.
  - Ensure examples and usage instructions are up to date.

---

## Submitting a Pull Request

1. **Sync with Upstream:** Ensure your fork is up-to-date with the main repository.
   ```bash
   git fetch upstream
   git merge upstream/main
   ```
2. **Submit a Pull Request:**
   - Go to your fork on GitHub.
   - Click on `Compare & pull request`.
   - Provide a detailed description of your changes.
3. **Address Feedback:** Be responsive to code review comments and make necessary updates.

---

## Getting Help

If you need assistance, feel free to:

- Post your question in the [Discussions](https://github.com/tinygrad/tinygrad/discussions) tab.
- Check the [FAQ](https://github.com/tinygrad/tinygrad/wiki/FAQ) or documentation.
- Tag maintainers or contributors in an issue for specific help.

---

Thank you for contributing to Tinygrad! Your efforts make this project better for everyone.

