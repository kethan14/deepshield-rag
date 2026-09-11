import { Link } from "@tanstack/react-router";
import { ArrowUpRight } from "lucide-react";

import { HoverLift, Reveal } from "@/components/motion-primitives";
import { Eyebrow, Shell, type RouteTo } from "@/components/site-ui";

const destinations: {
  to: RouteTo;
  name: string;
  description: string;
  ref: string;
}[] = [
  {
    to: "/assistant",
    name: "Policy Assistant",
    description: "The conversation workspace and its retrieved-source panel.",
    ref: "P-01",
  },
  {
    to: "/attack-lab",
    name: "Attack Lab",
    description: "Compose adversarial prompts and inspect the pipeline path.",
    ref: "P-02",
  },
  {
    to: "/experiments",
    name: "Experiments",
    description: "Configuration comparison and the reserved result surfaces.",
    ref: "P-03",
  },
  {
    to: "/evidence",
    name: "Evidence",
    description: "Filterable records of prompts, sources, and decisions.",
    ref: "P-04",
  },
];

export function ClosingSection() {
  return (
    <section className="border-b border-border bg-surface-alt" aria-labelledby="closing-heading">
      <Shell className="py-16 lg:py-24">
        <div className="grid gap-10 lg:grid-cols-12 lg:gap-12">
          <Reveal className="lg:col-span-4">
            <Eyebrow>05 — Continue</Eyebrow>
            <h2
              id="closing-heading"
              className="mt-4 font-display text-[26px] font-semibold leading-[1.2] tracking-tight text-foreground sm:text-[30px]"
            >
              Explore the research prototype.
            </h2>
            <p className="mt-4 max-w-sm text-[14.5px] leading-relaxed text-muted-foreground">
              Each workspace is an interface preview. The layouts are complete; the evaluation
              backend is not connected yet.
            </p>
          </Reveal>

          <div className="lg:col-span-8">
            <ul className="border-t border-border">
              {destinations.map((dest, i) => (
                <Reveal as="li" key={dest.to} delay={0.08 * i}>
                  <HoverLift>
                    <Link
                      to={dest.to}
                      className="group flex items-center gap-5 border-b border-border bg-transparent px-1 py-6 transition-colors hover:bg-surface"
                    >
                      <span className="label-mono w-12 shrink-0">{dest.ref}</span>
                      <span className="min-w-0 flex-1">
                        <span className="block font-display text-[17px] font-semibold tracking-tight text-foreground">
                          {dest.name}
                        </span>
                        <span className="mt-1 block text-[13px] leading-relaxed text-muted-foreground">
                          {dest.description}
                        </span>
                      </span>
                      <ArrowUpRight
                        className="h-[18px] w-[18px] shrink-0 text-muted-foreground transition-colors group-hover:text-foreground"
                        aria-hidden="true"
                      />
                    </Link>
                  </HoverLift>
                </Reveal>
              ))}
            </ul>
          </div>
        </div>
      </Shell>
    </section>
  );
}
