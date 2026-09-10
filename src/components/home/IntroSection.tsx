import { motion, useReducedMotion } from "framer-motion";

import { DrawnPath, Reveal } from "@/components/motion-primitives";
import { ActionLink, Eyebrow, Shell } from "@/components/site-ui";

const traits = ["Reproducible", "Evidence-driven", "Student-scale"];

export function IntroSection() {
  return (
    <section className="relative border-b border-border" aria-labelledby="intro-heading">
      <Shell className="grid gap-12 py-16 lg:grid-cols-12 lg:gap-10 lg:py-24">
        <div className="lg:col-span-5 lg:pr-4">
          <Reveal>
            <Eyebrow>Master&apos;s Capstone Research</Eyebrow>
            <p className="mt-6 font-display text-[13px] font-semibold tracking-tight text-primary">
              DeepShield-RAG
            </p>
            <h1
              id="intro-heading"
              className="mt-3 font-display text-[32px] font-semibold leading-[1.14] tracking-tight text-foreground sm:text-[40px]"
            >
              Understand how secure a RAG assistant really is.
            </h1>
            <p className="mt-5 max-w-md text-[15px] leading-relaxed text-muted-foreground">
              DeepShield-RAG is an evaluation platform for retrieval-augmented
              assistants. It runs structured adversarial prompts through layered
              guardrail configurations and records the retrieval, decision, and
              response evidence behind every outcome — so security and
              usefulness can be examined side by side.
            </p>
          </Reveal>

          <Reveal delay={0.1} className="mt-8 flex flex-wrap gap-3">
            <ActionLink to="/assistant">Explore the Assistant</ActionLink>
            <ActionLink to="/experiments" variant="secondary">
              View Research Design
            </ActionLink>
          </Reveal>

          <Reveal delay={0.18} className="mt-10 border-t border-border pt-5">
            <ul className="flex flex-wrap gap-x-8 gap-y-3">
              {traits.map((trait) => (
                <li key={trait} className="flex items-center gap-2">
                  <span
                    className="h-[5px] w-[5px] bg-[color:var(--line)]"
                    aria-hidden="true"
                  />
                  <span className="label-mono">{trait}</span>
                </li>
              ))}
            </ul>
          </Reveal>
        </div>

        <div className="lg:col-span-7">
          <PipelineDiagram />
        </div>
      </Shell>
    </section>
  );
}

const NODES = [
  { x: 20, y: 40, label: "User request", sub: "policy question" },
  { x: 20, y: 128, label: "Retrieval", sub: "government documents" },
  { x: 250, y: 128, label: "Guardrails", sub: "layered checks" },
  { x: 250, y: 216, label: "Generation", sub: "model response" },
  { x: 20, y: 216, label: "Evidence capture", sub: "traceable record" },
];

const LINK_PATH =
  "M110 68 L110 100 M110 156 L226 156 M334 156 L334 188 M226 244 L134 244";

export function PipelineDiagram() {
  const reduce = useReducedMotion();

  return (
    <Reveal
      as="figure"
      delay={0.08}
      className="relative overflow-hidden rounded-[14px] border border-border bg-surface p-5 shadow-[0_1px_2px_rgba(16,42,67,0.04),0_16px_40px_-28px_rgba(16,42,67,0.25)] sm:p-7"
    >
      <div className="flex items-center justify-between">
        <figcaption className="label-mono">Request path</figcaption>
        <span className="label-mono">Fig. 01</span>
      </div>

      <div className="relative mt-5">
        <svg
          viewBox="0 0 444 292"
          className="w-full"
          role="img"
          aria-label="Diagram: a user request flows through retrieval, guardrails, model generation, and evidence capture."
        >
          <g className="text-[color:var(--line)]">
            <DrawnPath d={LINK_PATH} strokeWidth={1} duration={1.6} />
            <DrawnPath
              d="M334 100 L334 128"
              strokeWidth={1}
              duration={0.8}
              delay={0.9}
            />
          </g>

          {/* return edge from guardrails back to retrieval, drawn faintly */}
          <g className="text-[color:var(--line)] opacity-70">
            <DrawnPath d="M226 172 L118 172" dashed strokeWidth={1} />
          </g>

          {NODES.map((node, i) => (
            <DiagramNode key={node.label} {...node} index={i} />
          ))}

          {!reduce && (
            <motion.circle
              r="3"
              fill="var(--primary)"
              initial={{ cx: 110, cy: 68, opacity: 0 }}
              animate={{
                cx: [110, 110, 226, 334, 334, 226, 134],
                cy: [68, 156, 156, 156, 244, 244, 244],
                opacity: [0, 1, 1, 1, 1, 1, 0],
              }}
              transition={{
                duration: 5.2,
                times: [0, 0.16, 0.34, 0.5, 0.68, 0.86, 1],
                repeat: Infinity,
                repeatDelay: 1.6,
                ease: "linear",
              }}
            />
          )}
        </svg>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
        <p className="text-[12.5px] text-muted-foreground">
          Each stage is inspectable; nothing is inferred from the model alone.
        </p>
        <span className="label-mono">Interface preview</span>
      </div>
    </Reveal>
  );
}

function DiagramNode({
  x,
  y,
  label,
  sub,
  index,
}: {
  x: number;
  y: number;
  label: string;
  sub: string;
  index: number;
}) {
  const reduce = useReducedMotion();
  const anim = reduce
    ? {}
    : {
        initial: { opacity: 0, y: 8 },
        whileInView: { opacity: 1, y: 0 },
        viewport: { once: true, margin: "-80px" },
        transition: { duration: 0.5, delay: 0.15 + index * 0.14 },
      };

  return (
    <motion.g {...anim}>
      <rect
        x={x}
        y={y}
        width="180"
        height="56"
        rx="10"
        fill="var(--surface)"
        stroke="var(--border)"
      />
      <rect x={x} y={y + 10} width="2" height="36" fill="var(--primary)" />
      <text
        x={x + 16}
        y={y + 24}
        className="font-display"
        fontSize="12"
        fontWeight="600"
        fill="var(--foreground)"
      >
        {label}
      </text>
      <text
        x={x + 16}
        y={y + 41}
        fontSize="10.5"
        fill="var(--muted-foreground)"
      >
        {sub}
      </text>
    </motion.g>
  );
}
