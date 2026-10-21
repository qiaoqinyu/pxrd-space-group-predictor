# Legacy container recipe

`Dockerfile.predict` is retained as an unvalidated historical prediction-only
recipe. It is not the supported installation path and does not provide the full
GPU workflow. Agents should use the tested native Conda installation in
[`environments/README.md`](../environments/README.md). Do not build this recipe
as part of the documented setup or smoke test.
