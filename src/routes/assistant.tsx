import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/policy-assistant")({
  head: () => ({
    meta: [
      { title: "Policy Assistant — DeepShield-RAG" },
      {
        name: "description",
        content:
          "F-1 student policy assistant demonstration for DeepShield-RAG.",
      },
      { property: "og:title", content: "Policy Assistant — DeepShield-RAG" },
      {
        property: "og:description",
        content:
          "F-1 student policy assistant demonstration for DeepShield-RAG.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: PolicyAssistantPage,
});

function PolicyAssistantPage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <h1 className="font-display text-3xl font-bold tracking-tight text-foreground">
        Policy Assistant
      </h1>
      <p className="mt-4 max-w-2xl text-muted-foreground">
        This page will host the F-1 student policy assistant interface. It is
        currently a placeholder for the frontend shell.
      </p>
    </main>
  );
}
