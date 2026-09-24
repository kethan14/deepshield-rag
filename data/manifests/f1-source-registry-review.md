# F-1 Policy Source Registry Review

Status: **approved scope and frozen version 1 source list — collection not yet started**

Review date: 2026-09-24

This document records the authoritative sources and scope for the first DeepShield-RAG F-1
policy corpus. The user delegated final curation to the project, and the list was frozen after
live verification of restricted official pages. Collection remains a separate milestone and
must use the validated machine-readable planned manifest.

## Proposed scope

The first corpus should answer general policy questions for F-1 students in U.S. colleges,
universities, seminaries, conservatories, and language-training programs. It should cover:

1. F-1 visa eligibility, application, admission, Form I-20, and SEVIS basics.
2. Maintaining status, full-course-of-study requirements, and program extensions.
3. On-campus employment and authorized off-campus employment.
4. Curricular Practical Training (CPT), Optional Practical Training (OPT), and STEM OPT.
5. School transfers, changes of education level, and reinstatement or change of status.
6. Travel, re-entry, program completion, departure, and grace periods.

The first corpus should exclude case-specific legal advice, M-1 and J-1 rules except when an
official source discusses them beside F-1 rules, F-2-dependent questions, school certification
operations, public elementary or secondary-school rules, tax advice, Social Security
procedures, country-specific visa reciprocity, and temporary Special Student Relief notices.
Those exclusions keep the benchmark coherent and prevent rapidly changing exceptions from
being presented as general rules.

## Authority and conflict policy

When sources overlap or appear to disagree, the system and ground-truth reviewers should use
this order:

1. Current eCFR regulatory text.
2. Current USCIS Policy Manual and current form instructions.
3. Current Department of State Foreign Affairs Manual for consular decisions.
4. Current ICE/SEVP operational guidance.
5. Current public guidance from Study in the States and Travel.State.Gov.

Every answer must cite the source actually supporting it. A lower-level explanatory source
must not override a regulation or controlling agency policy. Conflicts, ambiguous effective
dates, and superseded instructions must be flagged for human review rather than resolved by
the language model.

## Approved core sources

| ID                             | Authority                   | Official source                                                                                                                                                                           | Main coverage                                                                                                 | Collection review                                                                                 |
| ------------------------------ | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ecfr-8-cfr-214-2`             | Primary regulation          | [8 CFR 214.2 — Special requirements for admission, extension, and maintenance of status](https://www.ecfr.gov/current/title-8/chapter-I/subchapter-B/part-214/subpart-A/section-214.2)    | F-1 admission, full course of study, employment, CPT/OPT, transfer, reinstatement, duration and departure     | Include. Capture the current dated edition because the page reports a recent Title 8 amendment.   |
| `ecfr-8-cfr-274a-12`           | Primary regulation          | [8 CFR 274a.12 — Classes of aliens authorized to accept employment](https://www.ecfr.gov/current/title-8/chapter-I/subchapter-B/part-274a/subpart-B/section-274a.12)                      | Employment authorization categories and related conditions                                                    | Include. Verify the current dated edition at freeze time.                                         |
| `uscis-pm-f-chapter-2`         | Controlling agency policy   | [USCIS Policy Manual, Volume 2, Part F, Chapter 2 — Eligibility Requirements](https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-2)                                              | F-1 classification eligibility                                                                                | Include. Live page verified current as of 2026-09-23.                                             |
| `uscis-pm-f-chapter-3`         | Controlling agency policy   | [USCIS Policy Manual, Volume 2, Part F, Chapter 3 — Courses and Enrollment, Full Course of Study, and Reduced Course Load](https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-3) | Enrollment, full course of study, online study, and reduced course loads                                      | Include. Live page verified current as of 2026-09-23.                                             |
| `uscis-pm-f-chapter-4`         | Controlling agency policy   | [USCIS Policy Manual, Volume 2, Part F, Chapter 4 — School Transfer](https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-4)                                                       | Transfer eligibility and timing                                                                               | Include. Live page verified current as of 2026-09-23.                                             |
| `uscis-pm-f-chapter-5`         | Controlling agency policy   | [USCIS Policy Manual, Volume 2, Part F, Chapter 5 — Practical Training](https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-5)                                                    | CPT, OPT, STEM OPT, filing and training conditions                                                            | Include. Live page verified current as of 2026-09-23.                                             |
| `uscis-pm-f-chapter-6`         | Controlling agency policy   | [USCIS Policy Manual, Volume 2, Part F, Chapter 6 — Employment](https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-6)                                                            | On-campus and off-campus employment                                                                           | Include. Live page verified current as of 2026-09-23.                                             |
| `uscis-pm-f-chapter-7`         | Controlling agency policy   | [USCIS Policy Manual, Volume 2, Part F, Chapter 7 — Absences From the United States](https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-7)                                       | Temporary absences, study abroad, and return to study                                                         | Include. Live page verified current as of 2026-09-23.                                             |
| `uscis-pm-f-chapter-8`         | Controlling agency policy   | [USCIS Policy Manual, Volume 2, Part F, Chapter 8 — Change of Status, Extension of Stay, and Length of Stay](https://www.uscis.gov/policy-manual/volume-2-part-f-chapter-8)               | Change of status, extensions, reinstatement, and length of stay                                               | Include. Live page verified current as of 2026-09-23.                                             |
| `uscis-i-765-instructions`     | Current form instructions   | [Instructions for Form I-765, Application for Employment Authorization](https://www.uscis.gov/sites/default/files/document/forms/i-765instr.pdf)                                          | OPT, STEM OPT, and other F-1 employment-authorization filing requirements                                     | Include. Live USCIS page accepts the 08/21/25 edition as of the review date.                      |
| `uscis-i-539-instructions`     | Current form instructions   | [Instructions for Form I-539, Application to Extend/Change Nonimmigrant Status](https://www.uscis.gov/sites/default/files/document/forms/i-539instr.pdf)                                  | F-1 change of status and reinstatement evidence                                                               | Include. Live USCIS page accepts the 08/28/24 edition as of the review date.                      |
| `ice-sevis-employment`         | Agency operational guidance | [ICE/SEVP — Employment](https://www.ice.gov/sevis/employment)                                                                                                                             | On-campus work, hardship and emergent-circumstance employment, authorization limits, and DSO responsibilities | Include. It provides operational detail while citing the governing regulations.                   |
| `ice-sevis-practical-training` | Agency operational guidance | [ICE/SEVP — Practical Training](https://www.ice.gov/sevis/practical-training)                                                                                                             | CPT and OPT distinctions, eligibility, authorization, and reporting                                           | Include. Prefer the live HTML page over older practical-training infographics.                    |
| `ice-sevis-travel`             | Agency operational guidance | [ICE/SEVP — Travel](https://www.ice.gov/sevis/travel)                                                                                                                                     | Travel and re-entry documents for active students and students with pending or approved OPT                   | Include. Live official page verified during the review.                                           |
| `state-fam-9-402-5`            | Consular policy             | [9 FAM 402.5 — Students and Exchange Visitors: F, M, and J Visas](https://fam.state.gov/FAM/09FAM/09FAM040205.html)                                                                       | F-1 visa classification and consular adjudication                                                             | Include, but extract only F-1-relevant sections and retain section identifiers in chunk metadata. |
| `state-student-visa`           | Public consular guidance    | [Travel.State.Gov — Student Visa](https://travel.state.gov/content/travel/en/us-visas/study/student-visa.html)                                                                            | Visa application, required documents, interview, issuance, and entry basics                                   | Include as the user-facing explanation for consular topics.                                       |
| `sits-form-i-983-overview`     | Agency public guidance      | [Study in the States — Form I-983 Overview](https://studyinthestates.dhs.gov/form-i-983-overview)                                                                                         | STEM OPT training-plan requirements and responsibilities                                                      | Include. Live official page last updated 2025-10-02.                                              |

## Deferred or excluded candidates

| Candidate type                                            | Decision | Reason                                                                                                           |
| --------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------- |
| Old Study in the States transfer posts from 2013–2014     | Exclude  | The site itself warns that some content may be outdated; current regulation and policy sources are stronger.     |
| Older SEVP employment and practical-training infographics | Exclude  | They duplicate live HTML guidance and some contain superseded 17-month STEM OPT material.                        |
| Study in the States School Search results                 | Exclude  | This is a dynamic school directory, not a stable policy document.                                                |
| Public-school-specific F-1 guidance                       | Defer    | Public primary and secondary-school rules are outside the proposed higher-education scope.                       |
| Temporary Special Student Relief notices                  | Defer    | They are country- and time-specific and would make the initial benchmark unstable.                               |
| USCIS policy alerts that only describe earlier updates    | Defer    | The current Policy Manual should be collected instead; alerts are useful only for provenance or change analysis. |

## Coverage check

| Topic                                         | Proposed supporting sources                                                                                                     |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Visa eligibility and application              | 9 FAM 402.5; Travel.State.Gov Student Visa                                                                                      |
| Admission and maintaining status              | 8 CFR 214.2; USCIS Policy Manual Chapters 2, 3, and 8                                                                           |
| Full course of study and extensions           | 8 CFR 214.2; USCIS Policy Manual Chapters 3 and 8                                                                               |
| On-campus and hardship employment             | 8 CFR 214.2; 8 CFR 274a.12; USCIS Policy Manual Chapter 6; ICE Employment                                                       |
| CPT, OPT, and STEM OPT                        | 8 CFR 214.2; 8 CFR 274a.12; USCIS Policy Manual Chapter 5; ICE Practical Training; Form I-765 Instructions; Form I-983 Overview |
| Transfer, reinstatement, and change of status | 8 CFR 214.2; USCIS Policy Manual Chapters 4 and 8; Form I-539 Instructions                                                      |
| Travel and re-entry                           | USCIS Policy Manual Chapter 7; ICE Travel; Travel.State.Gov Student Visa; 9 FAM 402.5                                           |
| Completion, departure, and grace periods      | 8 CFR 214.2; USCIS Policy Manual Chapter 8                                                                                      |

## Freeze checklist

Before creating the machine-readable planned manifest, each included source must pass all of
the following checks:

- The canonical URL resolves on the approved government domain without an unsafe redirect.
- The live title and publishing agency match this review.
- The page or document is current on the freeze date; PDF edition dates are recorded.
- Media type, language, topic tags, and rights notes are recorded without guessing.
- The source adds needed authority or explanation rather than unnecessary duplication.
- Recent regulatory amendments have been compared with agency guidance for conflicts.
- The approved scope and final source list have received explicit human approval.

## Approval record

The user delegated the final choice to the project on 2026-09-24. The 17-source list above was
selected as the most effective student-scale corpus: broad enough to cover the defined topics,
but small enough for versioning, human review, and reproducible evaluation. Downloading and
hashing the source files remains a separate later step.
