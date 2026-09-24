Rebbe Yiddish ASR Corpus - data sheet export (2026-09-24)

corpus-datasheet.pdf        the report as printed to A4 (Chrome print engine)
corpus-datasheet.html       the same report as a single self-contained web page (open in any browser; light/dark aware; prints cleanly)
charts-svg/                 every chart as a standalone SVG (vector; drop into slides or docs)
charts-png/                 the same charts rasterized at 1600 px wide
data/dataset-stats.json     all corpus statistics the charts were drawn from
data/composition-by-farbrengen-year.csv   hours/clips/days/words by year of the original talk (Hebrew and secular years)
data/composition-by-publication-year.csv  by Daily Sicha publication year
data/splits.csv             train/dev/test composition and timing sources
data/test-results.csv       every evaluated model x decoder on the fixed dev/test days
data/learning-curves.csv    dev WER (200-clip subset) at every evaluation step of every training run

Numbers are computed from Sichos/asr/data/manifests/*.jsonl and Sichos/asr/reports/*; see the report's last section for the file behind each figure.
