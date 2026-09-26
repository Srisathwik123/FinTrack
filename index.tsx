import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Brain,
  Database,
  ExternalLink,
  Gift,
  Lightbulb,
  Lock,
  MessageSquareText,
  Pencil,
  Plus,
  Trash2,
  TrendingUp,
  Wand2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  CATEGORIES,
  CATEGORY_COLORS,
  HAND_LABELLED,
  SEED_DATASET,
  type Category,
} from "@/lib/dataset";
import { evaluate, trainModel, type Prediction } from "@/lib/nlp";
import { parseSms, SAMPLE_SMS } from "@/lib/upi";
import {
  buildSuggestions,
  categoryTotals,
  currentMonth,
  inr,
  loadBudgets,
  loadCorrections,
  loadExpenses,
  makeDemoExpenses,
  monthKey,
  monthlyTotals,
  saveBudgets,
  saveCorrections,
  saveExpenses,
  weeklyTotals,
  type Budgets,
  type Expense,
} from "@/lib/store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "FinTrack — ML Expense Categorizer for Students" },
      {
        name: "description",
        content:
          "FinTrack classifies student expenses into food, transport, education, bills and more using TF-IDF and Naive Bayes, fully offline, with budget alerts and monthly reports.",
      },
      { property: "og:title", content: "FinTrack — ML Expense Categorizer for Students" },
      {
        property: "og:description",
        content:
          "Privacy-preserving expense categorization demo: TF-IDF + Naive Bayes trained in your browser, UPI/SMS parsing, budget alerts and spending reports.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Home,
});

type Tab = "entry" | "reports" | "model" | "data";

const OFFER_LINKS = [
  { name: "Amazon", detail: "Coupons & daily deals", url: "https://www.amazon.in/coupons", mark: "A" },
  { name: "Flipkart", detail: "Offers store", url: "https://www.flipkart.com/offers-store", mark: "F" },
  { name: "PhonePe", detail: "Offers in your PhonePe app", url: "https://cms.phonepe.com/en/myhelp/payments-other-websites-and-apps/cashback-related-issues/how-can-i-apply-offer-payment-phonepe/", mark: "P" },
  { name: "Google Pay", detail: "Rewards in your GPay app", url: "https://pay.google.com/intl/en_in/about/", mark: "G" },
  { name: "Myntra", detail: "Fashion deals", url: "https://www.myntra.com/deals", mark: "M" },
  { name: "Swiggy", detail: "Food offers near you", url: "https://www.swiggy.com/offers-near-me", mark: "S" },
  { name: "Zomato", detail: "Browse restaurant offers", url: "https://www.zomato.com/", mark: "Z" },
] as const;

function OffersPanel() {
  return (
    <aside aria-label="Coupons and discounts" className="min-w-0 border-t border-border pt-7 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0">
      <div className="lg:sticky lg:top-6">
        <div className="flex items-center gap-2 text-primary">
          <Gift className="size-5" aria-hidden="true" />
          <h2 className="text-lg font-semibold text-foreground">Coupons & offers</h2>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">Explore discounts from your favourite apps.</p>
        <div className="mt-4 grid gap-x-6 sm:grid-cols-2 lg:grid-cols-1">
          {OFFER_LINKS.map((offer) => (
            <div key={offer.name} className="flex min-w-0 items-center gap-3 border-b border-border py-3 first:pt-0">
              <span aria-hidden="true" className="grid size-9 shrink-0 place-items-center rounded-md bg-secondary text-sm font-bold text-secondary-foreground">
                {offer.mark}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold">{offer.name}</p>
                <p className="text-xs leading-snug text-muted-foreground">{offer.detail}</p>
              </div>
              <Button asChild variant="ghost" size="icon" className="shrink-0 text-primary" title={`View ${offer.name} offers`}>
                <a href={offer.url} target="_blank" rel="noopener noreferrer" aria-label={`View ${offer.name} offers (opens in a new tab)`}>
                  <ExternalLink aria-hidden="true" />
                </a>
              </Button>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs leading-relaxed text-muted-foreground">Offers and codes vary by account and may change. Check terms on each site or app before paying.</p>
      </div>
    </aside>
  );
}

function Home() {
  const [expenses, setExpenses] = useState<Expense[]>([]);
  const [budgets, setBudgets] = useState<Budgets>(() => loadBudgets());
  const [corrections, setCorrections] = useState(() => loadCorrections());
  const [tab, setTab] = useState<Tab>("entry");
  const [ready, setReady] = useState(false);

  // hydrate from on-device storage (nothing leaves the browser)
  useEffect(() => {
    const stored = loadExpenses();
    setExpenses(stored.length ? stored : makeDemoExpenses());
    setBudgets(loadBudgets());
    setCorrections(loadCorrections());
    setReady(true);
  }, []);

  useEffect(() => {
    if (ready) saveExpenses(expenses);
  }, [expenses, ready]);

  // training set = seed dataset + the user's own corrections (online learning)
  const trainingSet = useMemo(() => [...SEED_DATASET, ...corrections], [corrections]);
  const { model } = useMemo(() => trainModel(trainingSet), [trainingSet]);
  // Honest evaluation: the hold-out test rows come only from the hand-labelled
  // data, while template variations stay in training.
  const metrics = useMemo(
    () =>
      evaluate(
        HAND_LABELLED,
        trainingSet.filter((s) => !HAND_LABELLED.includes(s)),
      ),
    [trainingSet],
  );

  const month = currentMonth();
  const monthExpenses = expenses.filter((e) => monthKey(e.date) === month);
  const prevMonth = (() => {
    const d = new Date();
    d.setDate(1);
    d.setMonth(d.getMonth() - 1);
    return d.toISOString().slice(0, 7);
  })();
  const prevExpenses = expenses.filter((e) => monthKey(e.date) === prevMonth);
  const suggestions = buildSuggestions(monthExpenses, budgets, prevExpenses);
  const monthTotal = monthExpenses.reduce((a, e) => a + e.amount, 0);
  const accuracyOnUserData = (() => {
    const judged = expenses.filter((e) => e.source !== "manual" || true);
    if (!judged.length) return 1;
    return judged.filter((e) => !e.corrected).length / judged.length;
  })();

  function addExpense(e: Expense) {
    setExpenses((prev) => [e, ...prev]);
  }

  function correct(id: string, category: Category) {
    setExpenses((prev) =>
      prev.map((e) =>
        e.id === id ? { ...e, category, corrected: e.predicted !== category } : e,
      ),
    );
    const target = expenses.find((e) => e.id === id);
    if (target && target.predicted !== category) {
      const next = [...corrections, { text: target.description, label: category }];
      setCorrections(next);
      saveCorrections(next);
    }
  }

  function remove(id: string) {
    setExpenses((prev) => prev.filter((e) => e.id !== id));
  }

  function updateBudget(c: Category, value: number) {
    const next = { ...budgets, [c]: value };
    setBudgets(next);
    saveBudgets(next);
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-border/70 bg-card/60 backdrop-blur">
         <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-5 py-5">
          <div className="grid size-11 place-items-center rounded-xl bg-primary text-primary-foreground">
            <Brain className="size-6" />
          </div>
          <div className="mr-auto">
             <h1 className="text-xl font-semibold">FinTrack</h1>
            <p className="text-sm text-muted-foreground">
              Privacy-preserving ML expense categorization for college students
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-border bg-secondary px-3 py-1.5 text-xs font-medium text-secondary-foreground">
            <Lock className="size-3.5" /> Runs 100% on-device
          </div>
        </div>
         <nav className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-5">
          {(
            [
              ["entry", "Add expense", Plus],
              ["reports", "Reports & advice", TrendingUp],
              ["model", "Model evaluation", Brain],
              ["data", "Dataset & pipeline", Database],
            ] as const
          ).map(([key, label, Icon]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`flex items-center gap-2 rounded-t-lg border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
                tab === key
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              <Icon className="size-4" />
              {label}
            </button>
          ))}
        </nav>
      </header>

       <main className="mx-auto max-w-7xl px-5 py-8">
        <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="This month" value={inr(monthTotal)} hint={`${monthExpenses.length} expenses`} />
          <Stat
            label="Top category"
            value={categoryTotals(monthExpenses).sort((a, b) => b.total - a.total)[0]?.category ?? "—"}
            hint={inr(categoryTotals(monthExpenses).sort((a, b) => b.total - a.total)[0]?.total ?? 0)}
          />
          <Stat
            label="Model accuracy (hold-out)"
            value={`${(metrics.accuracy * 100).toFixed(1)}%`}
            hint={`macro F1 ${(metrics.macroF1 * 100).toFixed(1)}%`}
          />
          <Stat
            label="Auto vs manual agreement"
            value={`${(accuracyOnUserData * 100).toFixed(0)}%`}
            hint={`${corrections.length} user corrections learned`}
          />
        </div>

         <div className="grid min-w-0 gap-8 lg:grid-cols-[minmax(0,1fr)_250px]">
           <div className="min-w-0">
             {tab === "entry" && (
               <EntryTab
                 model={model}
                 expenses={expenses}
                 onAdd={addExpense}
                 onCorrect={correct}
                 onRemove={remove}
               />
             )}
             {tab === "reports" && (
               <ReportsTab
                 expenses={expenses}
                 monthExpenses={monthExpenses}
                 month={month}
                 budgets={budgets}
                 onBudget={updateBudget}
                 suggestions={suggestions}
               />
             )}
             {tab === "model" && <ModelTab metrics={metrics} trainingSize={trainingSet.length} />}
             {tab === "data" && <DataTab corrections={corrections.length} />}
           </div>
           <OffersPanel />
         </div>
      </main>

      <footer className="border-t border-border/70 py-6 text-center text-xs text-muted-foreground">
         FinTrack demo — TF-IDF + Multinomial Naive Bayes trained in the browser. No financial
        data is sent anywhere.
      </footer>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="panel p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-2 text-2xl font-semibold font-display">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

/* ------------------------------ Entry tab ------------------------------ */

function EntryTab({
  model,
  expenses,
  onAdd,
  onCorrect,
  onRemove,
}: {
  model: ReturnType<typeof trainModel>["model"];
  expenses: Expense[];
  onAdd: (e: Expense) => void;
  onCorrect: (id: string, c: Category) => void;
  onRemove: (id: string) => void;
}) {
  const [description, setDescription] = useState("");
  const [amount, setAmount] = useState("");
  const [sms, setSms] = useState("");
  const [source, setSource] = useState<"manual" | "sms">("manual");

  const prediction: Prediction | null = description.trim().length > 2 ? model.predict(description) : null;

  function submit() {
    const amt = Number(amount);
    if (!description.trim() || !amt || amt <= 0) return;
    const p = model.predict(description);
    onAdd({
      id: crypto.randomUUID(),
      date: new Date().toISOString().slice(0, 10),
      description: description.trim(),
      amount: amt,
      predicted: p.label,
      confidence: p.confidence,
      category: p.label,
      corrected: false,
      source,
    });
    setDescription("");
    setAmount("");
    setSms("");
    setSource("manual");
  }

  function applySms(text: string) {
    setSms(text);
    const parsed = parseSms(text);
    setDescription(parsed.description);
    if (parsed.amount) setAmount(String(parsed.amount));
    setSource("sms");
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <section className="panel p-6">
        <h2 className="text-lg font-semibold">Enter an expense</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Type a note in plain language — the model predicts the category as you type.
        </p>

        <label className="mt-5 block text-sm font-medium">Description</label>
        <input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="e.g. Paid 120 to Swiggy using UPI"
          className="mt-2 w-full rounded-lg border border-input bg-background px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-ring/40"
        />

        <label className="mt-4 block text-sm font-medium">Amount (₹)</label>
        <input
          value={amount}
          onChange={(e) => setAmount(e.target.value.replace(/[^0-9.]/g, ""))}
          inputMode="decimal"
          placeholder="120"
          className="mt-2 w-full rounded-lg border border-input bg-background px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-ring/40"
        />

        {prediction && (
          <div className="mt-5 rounded-xl border border-border bg-secondary/60 p-4">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-sm text-muted-foreground">Predicted category</span>
              <span
                className="rounded-full px-3 py-1 text-sm font-semibold text-primary-foreground"
                style={{ backgroundColor: CATEGORY_COLORS[prediction.label] }}
              >
                {prediction.label}
              </span>
              <span className="text-sm font-medium">
                Confidence {(prediction.confidence * 100).toFixed(0)}%
              </span>
            </div>
            {prediction.topFeatures.length > 0 && (
              <p className="mt-2 text-xs text-muted-foreground">
                Decided from tokens: {prediction.topFeatures.map((f) => `"${f}"`).join(", ")}
              </p>
            )}
            <div className="mt-3 space-y-1.5">
              {prediction.scores.slice(0, 3).map((s) => (
                <div key={s.label} className="flex items-center gap-2 text-xs">
                  <span className="w-24 text-muted-foreground">{s.label}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${s.p * 100}%`, backgroundColor: CATEGORY_COLORS[s.label] }}
                    />
                  </div>
                  <span className="w-10 text-right tabular-nums">{(s.p * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          onClick={submit}
          className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
        >
          <Plus className="size-4" /> Save expense
        </button>

        <div className="mt-7 border-t border-border pt-5">
          <h3 className="flex items-center gap-2 text-sm font-semibold">
            <MessageSquareText className="size-4 text-primary" /> Paste a UPI / bank SMS
          </h3>
          <textarea
            value={sms}
            onChange={(e) => setSms(e.target.value)}
            onBlur={() => sms.trim() && applySms(sms)}
            rows={2}
            placeholder="Rs.120.00 debited from A/c XX4521 to SWIGGY via UPI Ref no 402318889231"
            className="mt-2 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
          />
          <div className="mt-2 flex flex-wrap gap-2">
            {SAMPLE_SMS.map((s) => (
              <button
                key={s}
                onClick={() => applySms(s)}
                className="rounded-full border border-border bg-background px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary hover:text-primary"
              >
                <Wand2 className="mr-1 inline size-3" />
                {s.slice(0, 34)}…
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="panel p-6">
        <h2 className="text-lg font-semibold">Recent expenses</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Wrong category? Correct it — the correction is added to the training data and the model
          retrains immediately.
        </p>
        <div className="mt-4 max-h-[32rem] space-y-2 overflow-y-auto pr-1">
          {expenses.slice(0, 40).map((e) => (
            <div key={e.id} className="rounded-xl border border-border p-3">
              <div className="flex items-start gap-3">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{e.description}</p>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {e.date} · {e.source === "sms" ? "from SMS" : "manual"} ·{" "}
                    {(e.confidence * 100).toFixed(0)}% confidence
                    {e.corrected && " · corrected"}
                  </p>
                </div>
                <span className="text-sm font-semibold tabular-nums">{inr(e.amount)}</span>
                <button
                  onClick={() => onRemove(e.id)}
                  aria-label="Delete expense"
                  className="text-muted-foreground transition-colors hover:text-destructive"
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <Pencil className="size-3.5 text-muted-foreground" />
                <select
                  value={e.category}
                  onChange={(ev) => onCorrect(e.id, ev.target.value as Category)}
                  className="rounded-md border border-input bg-background px-2 py-1 text-xs"
                >
                  {CATEGORIES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                {e.corrected && (
                  <span className="text-xs text-warning-foreground">
                    model said {e.predicted}
                  </span>
                )}
              </div>
            </div>
          ))}
          {!expenses.length && (
            <p className="py-8 text-center text-sm text-muted-foreground">No expenses yet.</p>
          )}
        </div>
      </section>
    </div>
  );
}

/* ------------------------------ Reports tab ------------------------------ */

function ReportsTab({
  expenses,
  monthExpenses,
  month,
  budgets,
  onBudget,
  suggestions,
}: {
  expenses: Expense[];
  monthExpenses: Expense[];
  month: string;
  budgets: Budgets;
  onBudget: (c: Category, v: number) => void;
  suggestions: ReturnType<typeof buildSuggestions>;
}) {
  const cats = categoryTotals(monthExpenses).filter((c) => c.total > 0);
  const months = monthlyTotals(expenses);
  const weeks = weeklyTotals(expenses, month);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="panel p-6">
          <h2 className="text-lg font-semibold">Category-wise spending ({month})</h2>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={cats} dataKey="total" nameKey="category" innerRadius={55} outerRadius={95}>
                  {cats.map((c) => (
                    <Cell key={c.category} fill={CATEGORY_COLORS[c.category]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => inr(v)} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex flex-wrap gap-3 text-xs">
            {cats.map((c) => (
              <span key={c.category} className="flex items-center gap-1.5">
                <span
                  className="size-2.5 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS[c.category] }}
                />
                {c.category} — {inr(c.total)}
              </span>
            ))}
          </div>
        </div>

        <div className="panel p-6">
          <h2 className="text-lg font-semibold">Weekly spending this month</h2>
          <div className="mt-4 h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weeks}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="week" tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
                <YAxis tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
                <Tooltip formatter={(v: number) => inr(v)} />
                <Bar dataKey="total" fill="var(--chart-1)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="panel p-6">
        <h2 className="text-lg font-semibold">Monthly trend</h2>
        <div className="mt-4 h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={months}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="month" tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
              <YAxis tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
              <Tooltip formatter={(v: number) => inr(v)} />
              <Line
                type="monotone"
                dataKey="total"
                stroke="var(--primary)"
                strokeWidth={2.5}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="panel p-6">
          <h2 className="text-lg font-semibold">Monthly student budget</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Edit any limit — alerts recalculate instantly.
          </p>
          <div className="mt-4 space-y-3">
            {CATEGORIES.map((c) => {
              const spent = monthExpenses
                .filter((e) => e.category === c)
                .reduce((a, e) => a + e.amount, 0);
              const pct = budgets[c] ? Math.min(100, (spent / budgets[c]) * 100) : 0;
              const over = budgets[c] && spent > budgets[c];
              return (
                <div key={c}>
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{c}</span>
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      {inr(spent)} of
                      <input
                        value={budgets[c]}
                        onChange={(e) => onBudget(c, Number(e.target.value.replace(/[^0-9]/g, "")) || 0)}
                        className="w-20 rounded-md border border-input bg-background px-2 py-1 text-right text-xs"
                      />
                    </span>
                  </div>
                  <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: over ? "var(--destructive)" : CATEGORY_COLORS[c],
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="panel p-6">
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Lightbulb className="size-5 text-accent" /> Budget suggestions
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Generated by simple rules on your own data — no external AI call.
          </p>
          <ul className="mt-4 space-y-2.5">
            {suggestions.map((s, i) => (
              <li
                key={i}
                className="rounded-xl border p-3 text-sm"
                style={{
                  borderColor:
                    s.level === "alert"
                      ? "var(--destructive)"
                      : s.level === "warn"
                        ? "var(--warning)"
                        : "var(--border)",
                }}
              >
                {s.text}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Model tab ------------------------------ */

function ModelTab({
  metrics,
  trainingSize,
}: {
  metrics: ReturnType<typeof evaluate>;
  trainingSize: number;
}) {
  return (
    <div className="space-y-6">
      <div className="panel p-6">
        <h2 className="text-lg font-semibold">Model evaluation (75 / 25 hold-out split)</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Multinomial Naive Bayes on TF-IDF features (unigrams, bigrams and character 4-grams). Test rows come only from the hand-labelled data. Training rows:{" "}
          {trainingSize} · train {metrics.trainSize} · test {metrics.testSize}.
        </p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase text-muted-foreground">
                <th className="py-2">Category</th>
                <th className="py-2 text-right">Precision</th>
                <th className="py-2 text-right">Recall</th>
                <th className="py-2 text-right">F1</th>
                <th className="py-2 text-right">Support</th>
              </tr>
            </thead>
            <tbody>
              {metrics.perClass.map((m) => (
                <tr key={m.label} className="border-b border-border/60">
                  <td className="py-2 font-medium">{m.label}</td>
                  <td className="py-2 text-right tabular-nums">{m.precision.toFixed(2)}</td>
                  <td className="py-2 text-right tabular-nums">{m.recall.toFixed(2)}</td>
                  <td className="py-2 text-right tabular-nums">{m.f1.toFixed(2)}</td>
                  <td className="py-2 text-right tabular-nums">{m.support}</td>
                </tr>
              ))}
              <tr className="font-semibold">
                <td className="py-2">Overall</td>
                <td className="py-2 text-right" colSpan={2}>
                  accuracy {(metrics.accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-2 text-right tabular-nums">{metrics.macroF1.toFixed(2)}</td>
                <td className="py-2 text-right tabular-nums">{metrics.testSize}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel p-6">
        <h2 className="text-lg font-semibold">Confusion matrix (test set)</h2>
        <div className="mt-4 overflow-x-auto">
          <table className="text-xs">
            <thead>
              <tr>
                <th className="p-2 text-left text-muted-foreground">actual ↓ / predicted →</th>
                {CATEGORIES.map((c) => (
                  <th key={c} className="p-2 text-muted-foreground">
                    {c.slice(0, 5)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.confusion.map((row, r) => (
                <tr key={r}>
                  <td className="p-2 font-medium">{CATEGORIES[r]}</td>
                  {row.map((v, c) => (
                    <td
                      key={c}
                      className="p-2 text-center tabular-nums"
                      style={{
                        backgroundColor: v
                          ? r === c
                            ? "oklch(0.58 0.13 165 / 0.18)"
                            : "oklch(0.577 0.19 25 / 0.18)"
                          : undefined,
                      }}
                    >
                      {v || ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          In the Python version this is the same report produced by
          <code className="mx-1 rounded bg-secondary px-1">classification_report</code> and
          <code className="mx-1 rounded bg-secondary px-1">confusion_matrix</code> from
          scikit-learn.
        </p>
      </div>
    </div>
  );
}

/* ------------------------------ Data tab ------------------------------ */

function DataTab({ corrections }: { corrections: number }) {
  const byCat = CATEGORIES.map((c) => ({
    category: c,
    count: SEED_DATASET.filter((s) => s.label === c).length,
  }));

  const steps = [
    ["1. Input", "User types a note, or pastes a UPI/bank SMS which is parsed for amount + merchant."],
    ["2. Cleaning", "Lowercase, remove punctuation/digits, drop stopwords, light stemming."],
    ["3. Vectorize", "TF-IDF over unigrams, bigrams and character 4-grams, L2 normalised."],
    ["4. Classify", "Multinomial Naive Bayes returns the category with a confidence score."],
    ["5. Store", "Expense saved locally (SQLite in the Python build, browser storage here)."],
    ["6. Correct & retrain", "A user correction becomes a new labelled row; the model refits."],
    ["7. Report & advise", "Charts plus rule-based budget alerts — no data leaves the device."],
  ];

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="panel p-6">
        <h2 className="text-lg font-semibold">Labelled dataset</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {HAND_LABELLED.length} hand-labelled rows + {SEED_DATASET.length - HAND_LABELLED.length}{" "}
          template variations + {corrections} rows learned from your corrections ={" "}
          {SEED_DATASET.length} training rows.
        </p>
        <div className="mt-4 h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={byCat} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
              <YAxis
                type="category"
                dataKey="category"
                width={90}
                tick={{ fontSize: 12 }}
                stroke="var(--muted-foreground)"
              />
              <Tooltip />
              <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                {byCat.map((b) => (
                  <Cell key={b.category} fill={CATEGORY_COLORS[b.category]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel p-6">
        <h2 className="text-lg font-semibold">Pipeline your team will build</h2>
        <ol className="mt-4 space-y-3">
          {steps.map(([title, body]) => (
            <li key={title} className="rounded-xl border border-border p-3">
              <p className="text-sm font-semibold">{title}</p>
              <p className="mt-0.5 text-sm text-muted-foreground">{body}</p>
            </li>
          ))}
        </ol>
        <div className="mt-5 rounded-xl bg-secondary/70 p-4 text-sm">
          <p className="font-semibold">Python equivalent</p>
          <pre className="mt-2 overflow-x-auto text-xs leading-relaxed text-muted-foreground">
{`TfidfVectorizer(ngram_range=(1,2), stop_words=...)
MultinomialNB()  # vs LogisticRegression()
train_test_split(X, y, test_size=0.25)
classification_report(y_test, y_pred)
sqlite3 -> expenses(id, date, desc, amount, predicted, final)
streamlit + plotly for the UI and charts`}
          </pre>
        </div>
      </div>
    </div>
  );
}
