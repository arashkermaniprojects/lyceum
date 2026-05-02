# Data-driven candidate ontology

Source: `triples.jsonl` (681 triples, 199 unique relation phrases).

Clustering: greedy single-link, cosine τ = 0.92, embeddings from Qwen2.5-7B mean-pool.

| # | total | representative | members |
|---|---|---|---|
| 1 | 79 | **used as** | used as (75), used in (2), used for (1), called (1) |
| 2 | 77 | **do not provide** | do not provide (72), does not do (1), might exist (1), are doing (1), need to find (1), are added (1) |
| 3 | 76 | **requires** | requires (76) |
| 4 | 75 | **require** | require (75) |
| 5 | 62 | **reduced** | reduced (62) |
| 6 | 34 | **is** | is (34) |
| 7 | 21 | **is like** | is like (3), is not (2), is composite (2), is greater than (2), is good (1), is long (1), is faster than (1), is likely (1), is different (1), is doing (1), is called (1), is of (1), is a (1), is either (1), is initially (1), is checking (1) |
| 8 | 11 | **will do** | will do (3), doesn't choose (1), can do (1), will go to (1), will compare (1), did not do (1), will go for (1), doesn't know (1), is going to (1) |
| 9 | 10 | **can be divided by** | can be divided by (2), can be applied to (1), can be ignored (1), are multiplied (1), cannot be less than (1), is used for (1), is used (1), will be used (1), is shown (1) |
| 10 | 10 | **costs** | costs (3), counted (1), multiplied (1), placed (1), check (1), overlaps (1), turns (1), pressed (1) |
| 11 | 10 | **equals** | equals (10) |
| 12 | 8 | **gives** | gives (8) |
| 13 | 8 | **uses** | uses (8) |
| 14 | 7 | **follows** | follows (7) |
| 15 | 7 | **is an example of** | is an example of (4), is a mix of (1), is the result of (1), is based on (1) |
| 16 | 5 | **do** | do (5) |
| 17 | 5 | **explains** | explains (5) |
| 18 | 5 | **sorts** | sorts (2), extracts (2), compresses (1) |
| 19 | 5 | **splits** | splits (2), sends (2), solves (1) |
| 20 | 4 | **across** | across (4) |
| 21 | 4 | **can be** | can be (2), will be (2) |
| 22 | 4 | **divided by** | divided by (4) |
| 23 | 4 | **reaches** | reaches (2), reduces efficiency (1), repeats (1) |
| 24 | 3 | **can't** | can't (2), can't be (1) |
| 25 | 3 | **chooses** | chooses (2), chose (1) |
| 26 | 3 | **compares** | compares (2), compared (1) |
| 27 | 3 | **described as** | described as (2), described by (1) |
| 28 | 3 | **has** | has (2), has list (1) |
| 29 | 3 | **presents** | presents (2), picks (1) |
| 30 | 3 | **shows** | shows (3) |
| 31 | 3 | **talks about** | talks about (3) |
| 32 | 3 | **thanks** | thanks (3) |
| 33 | 2 | **are** | are (2) |
| 34 | 2 | **asks** | asks (1), tracks (1) |
| 35 | 2 | **believes** | believes (2) |
| 36 | 2 | **build** | build (2) |
| 37 | 2 | **computes** | computes (1), computes fast (1) |
| 38 | 2 | **contains** | contains (2) |
| 39 | 2 | **does** | does (2) |
| 40 | 2 | **doing** | doing (2) |
| 41 | 2 | **ends with** | ends with (1), comes before (1) |
| 42 | 2 | **explain** | explain (2) |
| 43 | 2 | **get compared** | get compared (1), showed (1) |
| 44 | 2 | **goes through** | goes through (1), goes to (1) |
| 45 | 2 | **got** | got (2) |
| 46 | 2 | **has type** | has type (1), has complexity (1) |
| 47 | 2 | **have** | have (2) |
| 48 | 2 | **improves** | improves (2) |
| 49 | 2 | **includes** | includes (2) |
| 50 | 2 | **moves** | moves (1), visited (1) |
| 51 | 2 | **need** | need (2) |
| 52 | 2 | **reach** | reach (2) |
| 53 | 2 | **results in** | results in (2) |
| 54 | 2 | **said** | said (1), says (1) |
| 55 | 2 | **show** | show (2) |
| 56 | 2 | **try** | try (2) |
| 57 | 2 | **understand** | understand (2) |
| 58 | 2 | **use** | use (2) |
| 59 | 1 | **answer** | answer (1) |
| 60 | 1 | **are mentioned by** | are mentioned by (1) |
| 61 | 1 | **become** | become (1) |
| 62 | 1 | **becomes** | becomes (1) |
| 63 | 1 | **breaks down** | breaks down (1) |
| 64 | 1 | **can speed up** | can speed up (1) |
| 65 | 1 | **changes** | changes (1) |
| 66 | 1 | **choose** | choose (1) |
| 67 | 1 | **compute** | compute (1) |
| 68 | 1 | **converts to** | converts to (1) |
| 69 | 1 | **detected** | detected (1) |
| 70 | 1 | **did** | did (1) |
| 71 | 1 | **diverges** | diverges (1) |
| 72 | 1 | **examined** | examined (1) |
| 73 | 1 | **exists** | exists (1) |
| 74 | 1 | **exists in** | exists in (1) |
| 75 | 1 | **exploit** | exploit (1) |
| 76 | 1 | **explores** | explores (1) |
| 77 | 1 | **focuses on** | focuses on (1) |
| 78 | 1 | **gets padded** | gets padded (1) |
| 79 | 1 | **goes back to** | goes back to (1) |
| 80 | 1 | **had** | had (1) |
| 81 | 1 | **happens on** | happens on (1) |
| 82 | 1 | **helps** | helps (1) |
| 83 | 1 | **hit** | hit (1) |
| 84 | 1 | **implies** | implies (1) |
| 85 | 1 | **in** | in (1) |
| 86 | 1 | **involves** | involves (1) |
| 87 | 1 | **is divisible by** | is divisible by (1) |
| 88 | 1 | **lets you** | lets you (1) |
| 89 | 1 | **lose** | lose (1) |
| 90 | 1 | **makes** | makes (1) |
| 91 | 1 | **makes faster** | makes faster (1) |
| 92 | 1 | **means** | means (1) |
| 93 | 1 | **meant** | meant (1) |
| 94 | 1 | **mentioned by** | mentioned by (1) |
| 95 | 1 | **move to** | move to (1) |
| 96 | 1 | **moves on to** | moves on to (1) |
| 97 | 1 | **occurs in** | occurs in (1) |
| 98 | 1 | **part of** | part of (1) |
| 99 | 1 | **perceives** | perceives (1) |
| 100 | 1 | **points to** | points to (1) |
| 101 | 1 | **provides** | provides (1) |
| 102 | 1 | **put back into** | put back into (1) |
| 103 | 1 | **reduced by** | reduced by (1) |
| 104 | 1 | **relies on** | relies on (1) |
| 105 | 1 | **reminiscent of** | reminiscent of (1) |
| 106 | 1 | **repeat** | repeat (1) |
| 107 | 1 | **replace** | replace (1) |
| 108 | 1 | **response to** | response to (1) |
| 109 | 1 | **runs** | runs (1) |
| 110 | 1 | **section** | section (1) |
| 111 | 1 | **set to** | set to (1) |
| 112 | 1 | **signifies** | signifies (1) |
| 113 | 1 | **signify** | signify (1) |
| 114 | 1 | **similar to** | similar to (1) |
| 115 | 1 | **speeds up** | speeds up (1) |
| 116 | 1 | **start adding** | start adding (1) |
| 117 | 1 | **started** | started (1) |
| 118 | 1 | **stops** | stops (1) |
| 119 | 1 | **supports** | supports (1) |
| 120 | 1 | **swaps** | swaps (1) |
| 121 | 1 | **take** | take (1) |
| 122 | 1 | **teaching** | teaching (1) |
| 123 | 1 | **tells** | tells (1) |
| 124 | 1 | **terminates** | terminates (1) |
| 125 | 1 | **thank** | thank (1) |
| 126 | 1 | **want to send** | want to send (1) |
| 127 | 1 | **went** | went (1) |
| 128 | 1 | **works** | works (1) |
| 129 | 1 | **works on** | works on (1) |
