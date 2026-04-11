# Quarto Blog

Personal blog at [jtylerkirby.com](https://jtylerkirby.com), built with [Quarto](https://quarto.org/).

## Deployment

The site deploys automatically via GitHub Actions. Pushing to `main` triggers a workflow that renders the site with Quarto and publishes to the `gh-pages` branch, which GitHub Pages serves.

## Local Development

- `quarto preview` — Live preview with hot reload
- `quarto render` — Build the site to `_site/`
