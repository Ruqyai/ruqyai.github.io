# ruqyai.github.io

Personal site of **Ruqiya Bin Safi** (رقيا بن صافي): AI engineer and researcher working on generative AI, AI safety and Arabic NLP.

Live at [ruqyai.github.io](https://ruqyai.github.io) in English and [ruqyai.github.io/ar](https://ruqyai.github.io/ar/) in Arabic.

## How the site is organised

- `_data/home/en.yml` and `_data/home/ar.yml`: everything on the home page.
- `_posts/`, `_talks/`, `_teaching/`: English content. The same folders with `ar/` inside hold the Arabic content, served under `/ar/`.
- `_data/certificates.yml` and `images/certificates/`: the certificate gallery.
- `assets/css/site.css` (home page, top bar, footer) and `assets/css/skin.css` (inner pages): the design, with light and dark themes and right-to-left support.

## Run it locally

```bash
bundle install
bundle exec jekyll serve --config _config.yml,_config.dev.yml
```

Then open http://localhost:4000.

Built with Jekyll on GitHub Pages, from the AcademicPages and Minimal Mistakes themes.
