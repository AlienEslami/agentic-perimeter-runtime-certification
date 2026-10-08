# Reduction notes - 8 October 2026

The primary long manuscript is 77 pages. The reduced manuscript is 68 pages:
9 pages fewer, an 11.7% reduction with the same typography and page layout.
Source whitespace counts, which include LaTeX and table entries, fall from
21,555 to 18,374 (14.8%). These are editorial size comparisons.

## Mathematical and author-approved content

- Abstract and complete introduction: exact bytes preserved.
- Conclusion and future-work paragraph: unchanged.
- Four theorems, six propositions, one corollary and all eleven proofs:
  exact contents retained (excluding the removed optional heading explanations
  in the prior long version and whitespace in the reduced source).
- All 87 distinct displayed mathematical relations: retained. The long paper
  contains 89 displayed blocks because two allocation balances also appear
  in Appendix I; the reduced version retains those balances once in Appendix D
  and refers to them internally from I.
- Main numerical tables and four retained experimental appendix tables:
  numerical rows unchanged. Formal assumptions, terminal conditions, pending
  service coverage and finite-expiry limits remain explicit.

## Where the paper is shorter

1. Repeated explanations of rejection, retained correlated proofs and plan
   records in Section 4 and Appendices B/C are condensed.
2. Appendix F's prose repeating the terminal-backup theorem and proof is
   shortened; its statements, proof, conditions, equations and sustained-arrival
   restriction remain intact. Installation-timing exposition is condensed.
3. The operator discussion in Section 6.1 is shortened. The information and
   model-scope discussion is retained.
4. Appendices G-J retain essential protocols in a shorter account. Extended
   explanations refer to the corresponding appendices of the long paper.
5. G's delayed-handover table repeats the main timing table and is replaced
   by a reference. Its sequential traffic table provides fuller paired costs;
   the main results and complete six-case coverage table remain, with the
   additional paired reporting available in long G.7.
6. H's compatibility table gives extended primary-event metrics; the main
   eight-event summary and protocol remain, with those metrics available in
   long H.3. H's online figure repeats a comparison still reported in the
   retained table; the full figure and discussion remain in long H.2.
7. I points to Appendix D's allocation balances and derivative corrections.
   Its model, uncertainty grid, sixteen attempts and complexity formula remain.
8. J retains the agent snapshot, sample counts, interfaces, hard limits,
   objective weights, repair grid and replay procedure. Fuller histories,
   interfaces and scoring explanations remain in long J and repository records.

## Checks

The sources were compared independently and automatically. All references
resolve, result headings are plain, and final compilation reports no missing
citations, overfull boxes or oversized floats. All PDFs were rendered for
visual review, with detailed checks of result headings, changed pages and
experimental tables. The built-in compiler was unavailable on this platform;
the installed pdfLaTeX/BibTeX tools compiled the final manuscripts successfully.

The long source differs from the preceding version only by six removed
optional result titles and six preceding goal sentences. Numerical algorithms,
stored values and research records are unchanged. The repository stays private.

The later author-approved cleanup removes the funding declaration, updates
the writing/editing contributions, shortens the AI disclosure heading and
removes two editorial preparation notes. The change reduces the shortened
manuscript from 69 to 68 pages. The long sources and all mathematical content
are unchanged. Historical source packages retain their original declarations.
