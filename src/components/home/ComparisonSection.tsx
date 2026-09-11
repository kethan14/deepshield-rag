import { Reveal } from "@/components/motion-primitives";
import { SectionHeading, Shell } from "@/components/site-ui";

type Layer = { label: string; state: "present" | "partial" | "absent" };

const configurations: {
  id: string;
  name: string;
  intent: string;
  layers: Layer[];
}[] = [
  {
    id: "C0",
    name: "Basic RAG",
    intent:
      "A conventional retrieval-augmented assistant with no dedicated security layer. Serves as the reference path.",
    layers: [
      { label: "Retrieval", state: "present" },
      { label: "Prompt policy", state: "partial" },
      { label: "Retrieval inspection", state: "absent" },
      { label: "Guardrail layers", state: "absent" },
      { label: "Evidence record", state: "absent" },
    ],
  },
  {
    id: "C1",
    name: "Hardened RAG",
    intent:
      "The same pipeline with prompt hardening and basic output constraints applied at the model boundary.",
    layers: [
      { label: "Retrieval", state: "present" },
      { label: "Prompt policy", state: "present" },
      { label: "Retrieval inspection", state: "partial" },
      { label: "Guardrail layers", state: "partial" },
      { label: "Evidence record", state: "partial" },
    ],
  },
  {
    id: "C2",
    name: "Layered DeepShield",
    intent:
      "The full evaluation path, with risk analysis, retrieval inspection, layered guardrails, and complete evidence capture.",
    layers: [
      { label: "Retrieval", state: "present" },
      { label: "Prompt policy", state: "present" },
      { label: "Retrieval inspection", state: "present" },
      { label: "Guardrail layers", state: "present" },
      { label: "Evidence record", state: "present" },
    ],
  },
];

export function ComparisonSection() {
  return (
    <section className="border-b border-border bg-surface-alt" aria-labelledby="comparison-heading">
      <Shell className="py-16 lg:py-24">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <SectionHeading
            index="03"
            eyebrow="Controlled comparison"
            title="Three configurations of the same assistant."
            description="The comparison holds the documents, the model, and the questions constant, and varies only the defensive layers in the path."
          />
          <Reveal delay={0.1}>
            <p className="label-mono rounded-[10px] border border-dashed border-[color:var(--line)] px-3 py-2">
              Planned experimental configurations
            </p>
          </Reveal>
        </div>

        <div className="mt-14 space-y-4">
          {configurations.map((config, i) => (
            <Reveal
              key={config.id}
              delay={0.1 * i}
              className="rounded-[14px] border border-border bg-surface p-6 sm:p-7"
            >
              <div className="grid gap-6 lg:grid-cols-12 lg:items-center lg:gap-8">
                <div className="lg:col-span-4">
                  <div className="flex items-baseline gap-3">
                    <span className="font-mono text-[13px] font-medium text-primary">
                      {config.id}
                    </span>
                    <h3 className="font-display text-[19px] font-semibold tracking-tight text-foreground">
                      {config.name}
                    </h3>
                  </div>
                  <p className="mt-3 text-[13.5px] leading-relaxed text-muted-foreground">
                    {config.intent}
                  </p>
                </div>

                <div className="lg:col-span-8">
                  <ol className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
                    {config.layers.map((layer, index) => (
                      <li key={layer.label} className="relative flex-1 sm:min-w-0">
                        {index > 0 && (
                          <span
                            className="absolute -left-[6px] top-1/2 hidden h-[1px] w-[6px] bg-border sm:block"
                            aria-hidden="true"
                          />
                        )}
                        <LayerCell layer={layer} />
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.1} className="mt-6">
          <ul className="flex flex-wrap gap-x-7 gap-y-2">
            {(
              [
                ["present", "Layer included"],
                ["partial", "Layer partially applied"],
                ["absent", "Layer not applied"],
              ] as const
            ).map(([state, label]) => (
              <li key={state} className="flex items-center gap-2">
                <StateGlyph state={state} />
                <span className="label-mono">{label}</span>
              </li>
            ))}
          </ul>
        </Reveal>
      </Shell>
    </section>
  );
}

function LayerCell({ layer }: { layer: Layer }) {
  const styles = {
    present: {
      background: "var(--primary)",
      color: "#ffffff",
      borderColor: "var(--primary)",
    },
    partial: {
      background: "var(--surface-alt)",
      color: "var(--foreground)",
      borderColor: "var(--line)",
    },
    absent: {
      background: "transparent",
      color: "var(--muted-foreground)",
      borderColor: "var(--border)",
    },
  }[layer.state];

  return (
    <div
      className="flex h-full min-h-[58px] flex-col justify-center rounded-[10px] border px-3 py-2.5 transition-colors"
      style={{
        backgroundColor: styles.background,
        borderColor: styles.borderColor,
        borderStyle: layer.state === "absent" ? "dashed" : "solid",
      }}
    >
      <span
        className="font-display text-[12px] font-medium leading-snug"
        style={{ color: styles.color }}
      >
        {layer.label}
      </span>
    </div>
  );
}

function StateGlyph({ state }: { state: Layer["state"] }) {
  if (state === "present")
    return <span className="h-[10px] w-[10px] rounded-[3px] bg-primary" aria-hidden="true" />;
  if (state === "partial")
    return (
      <span
        className="h-[10px] w-[10px] rounded-[3px] border border-[color:var(--line)] bg-surface-alt"
        aria-hidden="true"
      />
    );
  return (
    <span
      className="h-[10px] w-[10px] rounded-[3px] border border-dashed border-border"
      aria-hidden="true"
    />
  );
}
