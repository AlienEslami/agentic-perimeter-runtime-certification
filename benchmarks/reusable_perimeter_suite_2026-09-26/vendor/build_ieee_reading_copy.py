"""Generate an IEEE reading source from the authoritative Elsevier manuscript.

Only document-class, front-matter, float, and bibliography formatting change.
Compile the emitted file with pdflatex/bibtex in the repository/package root.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'FOUNDATION_MFD_PARTB_ELSEVIER.tex').read_text(encoding='utf-8')
# Expand manuscript modules before applying layout-only transformations.
# The installed glyphtounicode input remains a TeX distribution dependency.
def expand_modules(text):
    pattern = r'\\input\{(manuscript_streamlined/[^}]+)\}'
    def read_module(match):
        path = ROOT / (match[1] if match[1].endswith('.tex') else match[1] + '.tex')
        return expand_modules(path.read_text(encoding='utf-8'))
    return re.sub(pattern, read_module, text)

source = expand_modules(source)
preamble, rest = source.split('\\begin{document}', 1)
front, body = rest.split('\\end{frontmatter}', 1)
preamble = preamble.replace('\\documentclass[preprint,12pt]{elsarticle}',
                            '\\documentclass[journal,10pt]{IEEEtran}\n\\usepackage[numbers,sort&compress]{natbib}')
preamble = re.sub(r'\\journal\{[^}]+\}', '', preamble)
preamble = preamble.replace('\\usepackage{lmodern}\n', '')
preamble = preamble.replace(r'\newcommand{\appref}[1]{\ref{#1}}',
                            r'\newcommand{\appref}[1]{Appendix~\ref{#1}}')
title = re.search(r'\\title\{([^}]+)\}', front).group(1)
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', front, re.S).group(1)
keywords = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', front, re.S).group(1)
keywords = ', '.join(part.strip() for part in keywords.split('\\sep'))
authors = re.findall(r'\\author\[[^]]+\]\{([^{}]+)(?:\\corref\{[^}]+\})?\}', front)
assert authors == ['Ali Eslami', 'Jiangbo Yu'], authors
front_ieee = r'''
\begin{document}
\title{%s}
\author{%s\thanks{The authors are with the Department of Civil Engineering,
McGill University, Montreal, Quebec, Canada (e-mail: ali.eslami@mcgill.ca;
jiangbo.yu@mcgill.ca).}\thanks{IEEE double-column reading copy of the
Transportation Research Part B manuscript.}}
\maketitle
\begin{abstract}%s\end{abstract}
\begin{IEEEkeywords}%s\end{IEEEkeywords}
''' % (title, ' and '.join(authors), abstract, keywords)
body = body.replace('\\appendix\n', '\\appendices\n\\numberwithin{equation}{section}\n')
body = body.replace('\\FloatBarrier\n', '').replace('\\clearpage\n', '')
body = body.replace('\\appendices\n', '\\FloatBarrier\n\\appendices\n')
body = body.replace('\\bibliographystyle', '\\FloatBarrier\n\\bibliographystyle')
body = body.replace('\\bibliographystyle{elsarticle-num}', '\\bibliographystyle{IEEEtranN}')
body = body.replace('\\scriptsize\\setlength{\\arraycolsep}{2.5pt}',
                    '\\tiny\\setlength{\\arraycolsep}{1.7pt}')
# Wide scientific figures and tables retain their original text-width scale.
body = re.sub(r'\\begin\{figure\}(\[[^]]*\])?(.*?)\\end\{figure\}',
              lambda m: ('\\begin{figure}[t]' + m[2] + '\\end{figure}')
              if '\\columnwidth' in m[2] else
              ('\\begin{figure*}[t]' + m[2] + '\\end{figure*}'), body, flags=re.S)
body = re.sub(r'\\begin\{table\}(\[[^]]*\])?(.*?)\\end\{table\}',
              lambda m: '\\begin{table*}[t]' + m[2] + '\\end{table*}', body, flags=re.S)
target = ROOT / 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY.tex'
target.write_text(preamble + front_ieee + body, encoding='utf-8')
print(target.name)
