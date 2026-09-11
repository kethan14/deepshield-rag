import { createFileRoute } from "@tanstack/react-router";
import { Route as RouteIcon, Layers } from "lucide-react";

import { Reveal } from "@/components/motion-primitives";
import {
  Disclaimer,
  EmptyState,
  PageIntro,
  Panel,
  PanelHeader,
  PrototypeNotice,
  Shell,
  StaticSelect,
} from "@/components/site-ui";

const title = "Attack Lab — DeepShield-RAG";
const description =
  "Interface preview of the DeepShield-RAG attack workspace: category selector, prompt editor, and pipeline inspection surface.";

export const Route = createFileRoute("/attack-lab")({
  head: () => ({
    meta: [
      { title },
      { name: "description", content: description },
      { property: "og:title", content: title },
      { property: "og:description", content: description },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: AttackLabPage,
});

const categories = [
  { id: "T-01", name: "Direct Injection" },
  { id: "T-02", name: "Indirect Injection" },
  { id: "T-03", name: "Prompt Extraction" },
  { id: "T-04", name: "Data Leakage" },
  { id: "T-05", name: "Citation Manipulation" },
  { id: "T-06", name: "Unauthorized Tool Use" },
];

const stages = [
  "Prompt",
  "Risk Analysis",
  "Retrieval Inspection",
  "Guardrail Execution",
  "Response Evaluation",
  "Evidence Record",
];

function AttackLabPage() {
  return (
    <main>
      <PageIntro
        eyebrow="Workspace P-02"
        title="Attack Lab"
        description="Compose adversarial prompts by category and follow them through each stage of the evaluation path."
        aside={
          <PrototypeNotice>
            Prototype interface — no attack is executed. Selections and prompt text are not sent
            anywhere.
          </PrototypeNotice>
        }
      />

      <Shell className="grid gap-6 py-12 lg:grid-cols-12 lg:py-16">
        <Reveal className="lg:col-span-4">
          <Panel>
            <PanelHeader title="Attack category" meta="6 planned" />
            <ul className="p-3">
              {categories.map((cat, i) => (
                <li key={cat.id}>
                  <button
                    type="button"
                    aria-pressed={i === 0}
                    className="flex w-full items-center gap-3 rounded-[10px] px-3 py-2.5 text-left transition-colors hover:bg-surface-alt"
                    style={i === 0 ? { backgroundColor: "var(--surface-alt)" } : undefined}
                  >
                    <span className="label-mono w-10 shrink-0">{cat.id}</span>
                    <span className="min-w-0 flex-1 truncate font-display text-[13.5px] font-medium text-foreground">
                      {cat.name}
                    </span>
                    <span
                      className="h-[7px] w-[7px] shrink-0 rounded-[2px] border border-[color:var(--line)]"
                      aria-hidden="true"
                    />
                  </button>
                </li>
              ))}
            </ul>
          </Panel>
        </Reveal>

        <div className="flex flex-col gap-6 lg:col-span-8">
          <Reveal delay={0.08}>
            <Panel>
              <PanelHeader title="Prompt editor" meta="T-01 · Direct Injection">
                <span className="label-mono">Draft</span>
              </PanelHeader>
              <div className="p-5">
                <label htmlFor="attack-prompt" className="sr-only">
                  Attack prompt
                </label>
                <textarea
                  id="attack-prompt"
                  rows={7}
                  className="w-full rounded-[10px] border border-border bg-surface-alt p-4 font-mono text-[13px] leading-relaxed text-foreground placeholder:text-muted-foreground"
                  placeholder="Write the adversarial prompt to evaluate…"
                />
                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                  <StaticSelect
                    id="attack-config"
                    label="Configuration"
                    value="C0 — Basic RAG"
                    options={["C0 — Basic RAG", "C1 — Hardened RAG", "C2 — Layered DeepShield"]}
                  />
                  <StaticSelect
                    id="attack-corpus"
                    label="Document set"
                    value="Public F-1 policy corpus"
                    options={["Public F-1 policy corpus"]}
                  />
                </div>
                <p className="label-mono mt-4">
                  Execution unavailable until the backend is connected
                </p>
              </div>
            </Panel>
          </Reveal>

          <Reveal delay={0.14}>
            <Panel>
              <PanelHeader title="Pipeline inspection">
                <Layers className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              </PanelHeader>
              <ol className="grid gap-px bg-border sm:grid-cols-3">
                {stages.map((stage, i) => (
                  <li key={stage} className="bg-surface p-4">
                    <div className="flex items-center justify-between">
                      <span className="label-mono">S{i + 1}</span>
                      <span
                        className="h-[7px] w-[7px] rounded-full border border-[color:var(--line)]"
                        aria-hidden="true"
                      />
                    </div>
                    <p className="mt-3 font-display text-[13px] font-medium text-foreground">
                      {stage}
                    </p>
                    <p className="mt-1 text-[12px] text-muted-foreground">Awaiting run</p>
                  </li>
                ))}
              </ol>
            </Panel>
          </Reveal>

          <Reveal delay={0.18}>
            <Panel>
              <PanelHeader title="Results" meta="0 runs" />
              <EmptyState
                icon={<RouteIcon className="h-5 w-5" aria-hidden="true" />}
                title="No runs recorded"
                description="When an attack is executed, its stage-by-stage trace and outcome will be listed here and written to the evidence log."
                className="py-14"
              />
            </Panel>
          </Reveal>
        </div>
      </Shell>

      <Disclaimer />
    </main>
  );
}
