# Phase 7 drafts vs the hanachos PDFs, by year (whole-file WER, scoring_text normalization)

Drafts: asr/output (third pass, turbo-full-lr3e5-4ep, GPU path; 011 18-Tishrei 5781 repaired 2026-09-23).
Reference: data/pdf_text/<year> (5781 clean text layer; 5778-5780 font-map decoded, bold font crib-pinned).

| year | days scored | ref words | WER (all words) | CER (all chars) | median day WER | days > 0.20 | no draft |
|---|---|---|---|---|---|---|---|
| 5778 | 247 | 319,404 | 0.0716 | 0.0407 | 0.059 | 10 | 0 |

5778 days above 0.20 WER (PDF/audio mismatch or truncated draft): 184 29-Sivan 0.30 (1372/1420 w), 049 15-Kisleiv 0.28 (1427/1429 w), 006 9-Tishrei 0.27 (175/188 w), 105 4-Adar 0.23 (1647/1679 w), 108 7-Adar 0.22 (1393/1359 w), 096 21-Shvat 0.22 (1592/1733 w), 145 2-Iyar 0.21 (1673/1735 w), 106 5-Adar 0.21 (1459/1465 w), 107 6-Adar 0.21 (1191/1174 w), 037 26-Cheshvan 0.20 (882/914 w)
| 5779 | 266 | 341,973 | 0.0698 | 0.0404 | 0.057 | 5 | 0 |

5779 days above 0.20 WER (PDF/audio mismatch or truncated draft): 143 2-Nisan 0.32 (1458/1439 w), 195 17-Sivan 0.30 (1554/1545 w), 125 5-Adar 2 0.26 (1842/1986 w), 126 6-Adar 2 0.23 (1327/1331 w), 082 4-Shvat 0.21 (1179/1298 w)
| 5780 | 245 | 306,533 | 0.0728 | 0.0419 | 0.057 | 10 | 0 |

5780 days above 0.20 WER (PDF/audio mismatch or truncated draft): 005 9-Tishrei 0.45 (396/369 w), 245 29-Elul 0.38 (1329/1061 w), 170 15-Sivan 0.27 (1208/1240 w), 084 8-Shvat 0.27 (850/983 w), 208 8-Av 0.26 (1559/1718 w), 131 14-Nisan 0.25 (141/165 w), 218 22-Av 0.21 (725/704 w), 110 14-Adar 0.21 (1463/1482 w), 235 17-Elul 0.21 (1486/1524 w), 209 9-Av 0.21 (1017/1033 w)
| 5781 | 247 | 312,740 | 0.0898 | 0.0497 | 0.071 | 9 | 0 |

5781 days above 0.20 WER (PDF/audio mismatch or truncated draft): 068 9-Teiveis 1.51 (1338/815 w), 053 17-Kisleiv 0.92 (1175/1543 w), 009 14-Tishrei 0.29 (1386/1555 w), 125 2-Nisan 0.28 (1279/1367 w), 151 9-Iyar 0.27 (1669/1700 w), 203 26-Tamuz 0.24 (1012/1113 w), 002 4-Tishrei 0.23 (1268/1286 w), 056 22-Kisleiv 0.21 (1680/1731 w), 011 18-Tishrei 0.20 (1394/1529 w)
