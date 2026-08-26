export const meta = {
  name: 'verify-one-lens',
  description: 'Adversarially verify one referee lens worth of findings against the paper, in a single batched agent',
  phases: [{ title: 'Verify', detail: 'one agent checks all of this lens’s findings' }],
}

// args = { lens: "identification", findings: [ {title, severity, location, claim, why_it_matters, already_addressed, fix}, ... ] }
// One agent per invocation. Nothing fans out. Worst case cost is one agent.

const ROOT = '/Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/writeup/draft_mert'

const lens = (args && args.lens) || 'unknown'
const findings = (args && args.findings) || []

if (!findings.length) {
  return { lens, error: 'no findings passed in args.findings' }
}

const VERDICTS_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          survives: { type: 'boolean' },
          revised_severity: { type: 'string', enum: ['fatal', 'major', 'moderate', 'minor', 'dropped'] },
          verdict_reason: { type: 'string', description: 'Quote the text that settles it, with file and line.' },
          already_answered_where: { type: 'string', description: 'file + section if the paper already answers it, else empty' },
          correction: { type: 'string', description: 'if the finding misstates the paper, the correction; else empty' },
          sharpened_claim: { type: 'string', description: 'the surviving criticism in its most defensible form' },
        },
        required: ['title', 'survives', 'revised_severity', 'verdict_reason', 'sharpened_claim'],
      },
    },
  },
  required: ['verdicts'],
}

const list = findings.map((f, i) => `
--- FINDING ${i + 1} ---
Title: ${f.title}
Severity claimed: ${f.severity}
Location: ${f.location}
Claim: ${f.claim}
Why it matters: ${f.why_it_matters}
Referee's note on what the paper already does: ${f.already_addressed || '(none)'}
`).join('\n')

phase('Verify')

const res = await agent(`You are an ADVERSARIAL VERIFIER for a paper being prepared for the Review of Economic Studies.

PAPER: "Chaining Tasks, Redefining Work: A Theory of AI Automation"
LaTeX source: ${ROOT}   (read with Bash: cat, sed -n, grep)
Files: 0_main.tex (abstract), 1_introduction.tex, 2_literature.tex, 3_shortrun.tex (model),
4_implications.tex (Prop 1 overturning, Prop 2 fragmentation, 4.3 non-monotonicity),
5_longrun.tex (jobs/wages/hand-offs), 6_extensions.tex (CES + DP algorithms), 7_empirics.tex,
8_conclusion.tex, A_omitted_proofs.tex, B_macro_production.tex, C_sample_construction.tex,
D_tables_and_robustness.tex, E_gpt_prompts.tex, F_prompt_robustness.tex,
G_frequency_robustness.tex, H_external_validation.tex.
Data and code, if a claim needs checking numerically:
  /Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/data/computed_objects/
  /Users/peymansh/MIT Dropbox/Peyman Shahidi/GitHub/rubin/analysis/

Another referee, working the "${lens}" lens, raised the ${findings.length} criticisms below.
Your job is to REFUTE them where you can. Referee agents routinely misremember the text, invent
quotations, and demand exercises the appendices already contain.

For EACH finding, in order:
 1. Is it factually accurate about what the paper says? Quote the actual line (file + line number).
    If the paper does not say what the criticism claims, mark survives=false and give the correction.
 2. Does the paper already answer it anywhere, including footnotes and all eight appendices? Grep hard.
    Fully answered -> dropped. Partially -> downgrade and say exactly what remains.
 3. Is the criticism correct on the economics, econometrics, or mathematics, not merely plausible?
    If it rests on a technical claim ("this is mechanical", "not identified", "the proof is wrong"),
    work it out yourself. Where the claim is numerical and the data are available, check it.
 4. Is the severity right for ReStud specifically? Default to downgrading when uncertain.

Then restate each survivor in the sharpest form the text actually supports.
Return one verdict per finding, in the same order. Be concise: this output is read by a co-author.

${list}`, {
  label: `verify:${lens}`,
  phase: 'Verify',
  schema: VERDICTS_SCHEMA,
  effort: 'high',
})

if (!res || !res.verdicts) return { lens, error: 'verifier returned nothing' }

const kept = res.verdicts.filter((v) => v.survives && v.revised_severity !== 'dropped')
log(`${lens}: ${res.verdicts.length} checked, ${kept.length} survived`)

return {
  lens,
  checked: res.verdicts.length,
  survived: kept.length,
  verdicts: res.verdicts,
}
