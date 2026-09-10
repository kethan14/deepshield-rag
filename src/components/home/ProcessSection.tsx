import { useRef, useState } from "react";
import {
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
} from "framer-motion";

import { Reveal } from "@/components/motion-primitives";
import { Eyebrow, Shell } from "@/components/site-ui";

const stages = [
  {
    id: "S1",
    name: "Prompt",
    summary: "The request enters the pipeline exactly as written.",
    detail:
      "Every submission is stored verbatim with its configuration, so a run can be repeated later under identical conditions.",
  },
  {
    id: "S2",
    name: "Risk Analysis",
    summary: "The request is classified before anything is retrieved.",
    detail:
      "Intent, sensitivity, and known adversarial patterns are examined first, which decides how strictly the later stages behave.",
  },
  {
    id: "S3",
    name: "Retrieval Inspection",
    summary: "Retrieved passages are checked, not trusted.",
    detail:
      "Source provenance and embedded instructions inside the retrieved text are inspected separately from the user request.",
  },
  {
    id: "S4",
    name: "Guardrail Execution",
    summary: "Layered policy checks run against the assembled context.",
    detail:
      "Each guardrail layer can be enabled or disabled independently, which is what makes configuration comparison meaningful.",
  },
  {
    id: "S5",
    name: "Response Evaluation",
    summary: "The answer is judged on grounding and usefulness.",
    detail:
      "A response is examined for support in the cited passages and for whether it still answers the question that was asked.",
  },
  {
    id: "S6",
    name: "Evidence Record",
    summary: "The full trace is written down for review.",
    detail:
      "Prompt, retrieved sources, guardrail decisions, and final response are stored together as one reviewable record.",
  },
];

export function ProcessSection() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [active, setActive] = useState(0);
  const reduce = useReducedMotion();

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start center", "end center"],
  });

  useMotionValueEvent(scrollYProgress, "change", (v) => {
    const next = Math.min(
      stages.length - 1,
      Math.max(0, Math.floor(v * stages.length))
    );
    setActive(next);
  });

  const current = stages[active] ?? stages[0]!;

  return (
    <section className="border-b border-border" aria-labelledby="process-heading">
      <Shell className="py-16 lg:py-24">
        <Reveal className="max-w-2xl">
          <div className="flex items-center gap-3">
            <span className="label-mono text-foreground/70">02</span>
            <span className="h-[1px] w-8 bg-border" aria-hidden="true" />
            <Eyebrow>The DeepShield process</Eyebrow>
          </div>
          <h2
            id="process-heading"
            className="mt-4 font-display text-[26px] font-semibold leading-[1.2] text-foreground sm:text-[32px]"
          >
            Six stages, each one recorded.
          </h2>
          <p className="mt-4 text-[15px] leading-relaxed text-muted-foreground">
            Evaluation is not a single verdict. A request is carried through six
            stages, and each stage produces something that can be inspected on
            its own.
          </p>
        </Reveal>

        <div
          ref={containerRef}
          className="mt-12 grid gap-10 lg:grid-cols-12 lg:gap-12"
        >
          {/* sticky explanation */}
          <div className="lg:col-span-5">
            <div className="lg:sticky lg:top-28">
              <div className="rounded-[14px] border border-border bg-surface p-6 shadow-[0_1px_2px_rgba(16,42,67,0.04)]">
                <div className="flex items-center justify-between">
                  <span className="label-mono">Stage {active + 1} of 6</span>
                  <span className="label-mono text-foreground/70">
                    {current.id}
                  </span>
                </div>

                <motion.div
                  key={current.id}
                  initial={reduce ? undefined : { opacity: 0, y: 12 }}
                  animate={reduce ? undefined : { opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: [0.22, 0.61, 0.36, 1] }}
                >
                  <h3 className="mt-5 font-display text-[22px] font-semibold tracking-tight text-foreground">
                    {current.name}
                  </h3>
                  <p className="mt-3 text-[14.5px] leading-relaxed text-foreground/80">
                    {current.summary}
                  </p>
                  <p className="mt-3 text-[13.5px] leading-relaxed text-muted-foreground">
                    {current.detail}
                  </p>
                </motion.div>

                <div
                  className="mt-6 flex gap-1.5 border-t border-border pt-5"
                  aria-hidden="true"
                >
                  {stages.map((s, i) => (
                    <span
                      key={s.id}
                      className="h-[3px] flex-1 rounded-full transition-colors duration-300"
                      style={{
                        backgroundColor:
                          i <= active ? "var(--primary)" : "var(--border)",
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* stage track */}
          <ol className="lg:col-span-7">
            {stages.map((stage, i) => {
              const isActive = i === active;
              const isPast = i < active;
              return (
                <li key={stage.id} className="relative pl-10 sm:pl-14">
                  <span
                    className="absolute left-[11px] top-0 h-full w-[1px] sm:left-[15px]"
                    style={{
                      backgroundColor: isPast
                        ? "var(--primary)"
                        : "var(--border)",
                    }}
                    aria-hidden="true"
                  />
                  <span
                    className="absolute left-0 top-[26px] grid h-[23px] w-[23px] place-items-center rounded-full border transition-colors duration-300 sm:left-1 sm:h-[23px] sm:w-[23px]"
                    style={{
                      borderColor: isActive
                        ? "var(--primary)"
                        : "var(--border)",
                      backgroundColor: isActive
                        ? "var(--primary)"
                        : "var(--surface)",
                    }}
                    aria-hidden="true"
                  >
                    <span
                      className="font-mono text-[10px]"
                      style={{
                        color: isActive
                          ? "var(--primary-foreground)"
                          : "var(--muted-foreground)",
                      }}
                    >
                      {i + 1}
                    </span>
                  </span>

                  <div
                    className="border-t border-border py-6 transition-opacity duration-300"
                    style={{ opacity: isActive ? 1 : 0.55 }}
                  >
                    <h3
                      className="font-display text-[17px] font-semibold tracking-tight transition-colors duration-300"
                      style={{
                        color: isActive
                          ? "var(--foreground)"
                          : "var(--muted-foreground)",
                      }}
                    >
                      {stage.name}
                    </h3>
                    <p className="mt-2 max-w-lg text-[13.5px] leading-relaxed text-muted-foreground">
                      {stage.summary}
                    </p>
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      </Shell>
    </section>
  );
}
