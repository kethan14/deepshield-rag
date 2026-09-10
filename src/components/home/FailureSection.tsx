import { DrawnPath, Reveal } from "@/components/motion-primitives";
import { SectionHeading, Shell } from "@/components/site-ui";

const paths = [
  {
    id: "F-01",
    title: "Manipulated instructions",
    body: "Retrieved text carries hidden directions, and the assistant follows the document instead of the policy it was given.",
  },
  {
    id: "F-02",
    title: "Unsupported answers",
    body: "A confident response is produced without grounding in the retrieved passages, or with citations that do not support it.",
  },
  {
    id: "F-03",
    title: "Sensitive information exposure",
    body: "System instructions, internal context, or restricted document content surface in the reply to an ordinary-looking question.",
  },
];

export function FailureSection() {
  return (
    <section
      className="border-b border-border bg-surface-alt"
      aria-labelledby="failure-heading"
    >
      <Shell className="py-16 lg:py-24">
        <SectionHeading
          index="01"
          eyebrow="Where the path breaks"
          title="The same request can fail in three different ways."
          description="A retrieval-augmented assistant does not fail like a search engine. The failure happens inside the path — between the document it reads, the rules it was given, and the answer it writes."
        />

        <div className="relative mt-14 lg:mt-16">
          {/* continuous line carried down from the request path diagram */}
          <svg
            className="pointer-events-none absolute left-0 top-0 hidden h-full w-[120px] text-[color:var(--line)] lg:block"
            viewBox="0 0 120 460"
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            <DrawnPath d="M40 -60 L40 92" strokeWidth={1} duration={0.8} />
            <DrawnPath
              d="M40 92 C40 150 118 130 118 190"
              strokeWidth={1}
              duration={1}
              delay={0.2}
            />
            <DrawnPath
              d="M40 92 C40 220 118 250 118 320"
              strokeWidth={1}
              duration={1.1}
              delay={0.35}
            />
            <DrawnPath
              d="M40 92 C40 300 118 360 118 440"
              strokeWidth={1}
              duration={1.2}
              delay={0.5}
            />
          </svg>

          <ol className="space-y-0 lg:pl-[120px]">
            {paths.map((path, i) => (
              <Reveal
                as="li"
                key={path.id}
                delay={0.12 * i}
                className="grid gap-4 border-t border-border py-8 sm:grid-cols-[96px_1fr] sm:gap-8 lg:py-10"
              >
                <span className="label-mono pt-1">{path.id}</span>
                <div className="max-w-2xl">
                  <h3 className="font-display text-[19px] font-semibold tracking-tight text-foreground">
                    {path.title}
                  </h3>
                  <p className="mt-3 text-[14.5px] leading-relaxed text-muted-foreground">
                    {path.body}
                  </p>
                </div>
              </Reveal>
            ))}
          </ol>
        </div>
      </Shell>
    </section>
  );
}
