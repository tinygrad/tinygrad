# Agent Notes

## Testing

Always run pytest with `-n12` for parallel execution:

```
python -m pytest -n12
```

## Lint and Typecheck

- Lint: `python3 -m ruff check .`
- Typecheck: `python3 -m mypy`
