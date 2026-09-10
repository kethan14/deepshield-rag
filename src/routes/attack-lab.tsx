import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/attack-lab")({
  head: () => ({
    meta: [
      { title: "Attack Lab — DeepShield-RAG" },
      {
        name: "description",
        content:
          "Attack testing workspace for DeepShield-RAG.",
      },
      { property: "og:title", content: "Attack Lab — DeepShield-RAG" },
      {
        property: "og:description",
        content:
          "Attack testing workspace for DeepShield-RAG.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: AttackLabPage,
});

function AttackLabPage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <h1 className="font-display text-3xl font-bold tracking-tight text-foreground">
        Attack Lab
      </h1>
      <p className="mt-4 max-w-2xl text-muted-foreground">
        This page will host the attack testing workspace. It is currently a
        placeholder for the frontend shell.
      </p>
    </main>
  );
}
