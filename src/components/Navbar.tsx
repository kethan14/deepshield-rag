import { useEffect, useState } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Menu, X } from "lucide-react";

const navItems = [
  { label: "Overview", to: "/" as const },
  { label: "Policy Assistant", to: "/assistant" as const },
  { label: "Attack Lab", to: "/attack-lab" as const },
  { label: "Experiments", to: "/experiments" as const },
  { label: "Evidence", to: "/evidence" as const },
];

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const reduce = useReducedMotion();

  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  const isActive = (to: string) =>
    to === "/" ? pathname === "/" : pathname.startsWith(to);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/90 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-[1200px] items-center justify-between px-6 lg:px-10">
        <Link
          to="/"
          className="group flex items-center gap-3"
          aria-label="DeepShield-RAG home"
        >
          <Mark />
          <span className="flex flex-col leading-none">
            <span className="font-display text-[15px] font-semibold tracking-tight text-foreground">
              DeepShield-RAG
            </span>
            <span className="label-mono mt-1 text-[10px]">
              Capstone Research
            </span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex" aria-label="Primary">
          {navItems.map((item) => {
            const active = isActive(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                className="relative px-3 py-2 text-[13px] font-medium transition-colors"
                style={{
                  color: active ? "var(--foreground)" : "var(--muted-foreground)",
                }}
              >
                <span className="relative z-10 hover:text-foreground">
                  {item.label}
                </span>
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-x-2 -bottom-[1px] h-[2px] bg-primary"
                    transition={
                      reduce
                        ? { duration: 0 }
                        : { duration: 0.32, ease: [0.22, 0.61, 0.36, 1] }
                    }
                  />
                )}
              </Link>
            );
          })}
        </nav>

        <button
          type="button"
          className="inline-flex h-10 w-10 items-center justify-center rounded-[10px] border border-border bg-surface text-foreground transition-colors hover:bg-surface-alt md:hidden"
          onClick={() => setIsOpen((v) => !v)}
          aria-expanded={isOpen}
          aria-controls="mobile-nav"
          aria-label={isOpen ? "Close navigation" : "Open navigation"}
        >
          {isOpen ? (
            <X className="h-[18px] w-[18px]" aria-hidden="true" />
          ) : (
            <Menu className="h-[18px] w-[18px]" aria-hidden="true" />
          )}
        </button>
      </div>

      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            id="mobile-nav"
            className="overflow-hidden border-t border-border bg-surface md:hidden"
            {...(reduce
              ? {}
              : {
                  initial: { height: 0, opacity: 0 },
                  animate: { height: "auto", opacity: 1 },
                  exit: { height: 0, opacity: 0 },
                  transition: {
                    duration: 0.28,
                    ease: [0.22, 0.61, 0.36, 1] as const,
                  },
                })}
          >
            <nav
              className="mx-auto max-w-[1200px] px-6 py-3"
              aria-label="Mobile"
            >
              <ul className="divide-y divide-border">
                {navItems.map((item, i) => {
                  const active = isActive(item.to);
                  return (
                    <li key={item.to}>
                      <Link
                        to={item.to}
                        className="flex items-center justify-between py-3 text-sm font-medium"
                        style={{
                          color: active
                            ? "var(--foreground)"
                            : "var(--muted-foreground)",
                        }}
                      >
                        {item.label}
                        <span className="label-mono text-[10px]">
                          {String(i).padStart(2, "0")}
                        </span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}

function Mark() {
  return (
    <span
      className="grid h-9 w-9 shrink-0 place-items-center rounded-[10px] border border-border bg-surface"
      aria-hidden="true"
    >
      <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]">
        <rect
          x="4.5"
          y="4.5"
          width="15"
          height="15"
          rx="2"
          fill="none"
          stroke="var(--line)"
        />
        <path d="M8 12h8M12 8v8" stroke="var(--line)" strokeWidth="1" />
        <rect x="9.5" y="9.5" width="5" height="5" fill="var(--primary)" />
      </svg>
    </span>
  );
}
