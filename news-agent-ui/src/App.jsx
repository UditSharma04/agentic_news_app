import AskAIPage from "./AskAIPage";

import { Menu } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useEffect, useMemo, useState } from "react";

function formatDate(d) {
  if (!d) return "";
  return new Date(d).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

import { MessageCircle } from "lucide-react";
import {
  LayoutDashboard,
  Newspaper,
  Mail,
  Globe,
  History,
  Search,
  Filter,
  Send,
  Sparkles,
  ChevronRight,
  ExternalLink,
  Moon,
  Sun,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";


import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

import { format } from "date-fns";

// ------------------------------------------------------------
// Dummy data (later we will replace with FastAPI backend)
// ------------------------------------------------------------
const DUMMY_SOURCES = [
  // NewsAPI
  { id: "reuters", name: "Reuters", type: "NewsAPI", status: "working" },

  // RSS
  { id: "bbc", name: "BBC", type: "RSS", status: "working" },
  { id: "techcrunch", name: "TechCrunch", type: "RSS", status: "working" },
  { id: "theverge", name: "The Verge", type: "RSS", status: "working" },
  { id: "arstechnica", name: "Ars Technica", type: "RSS", status: "working" },
  { id: "wired", name: "Wired", type: "RSS", status: "working" },
  { id: "mittechreview", name: "MIT Technology Review", type: "RSS", status: "working" },
  { id: "venturebeat", name: "VentureBeat", type: "RSS", status: "working" },
  { id: "theregister", name: "The Register", type: "RSS", status: "working" },
  { id: "hackernews", name: "Hacker News", type: "RSS", status: "working" },
];


const DUMMY_ARTICLES = [
  {


    id: "a1",
    title: "Global markets react to new inflation data",
    source: "Reuters",
    publishedAt: "2026-02-13T09:10:00Z",
    tags: ["Business", "Markets"],
    summary:
      "Major stock indexes moved sharply after new inflation data signaled changing interest-rate expectations.",
    url: "https://example.com/1",
    coverageCount: 4,
    isDuplicateGroup: true,
  },
  {
    id: "a2",
    title: "AI regulation talks accelerate across major economies",
    source: "BBC",
    publishedAt: "2026-02-13T12:40:00Z",
    tags: ["Tech", "AI", "Policy"],
    summary:
      "Governments are aligning on baseline rules for high-risk AI systems, focusing on transparency and safety.",
    url: "https://example.com/2",
    coverageCount: 6,
    isDuplicateGroup: true,
  },
  {
    id: "a3",
    title: "New cybersecurity breach highlights risks of personal devices",
    source: "NDTV",
    publishedAt: "2026-02-13T14:20:00Z",
    tags: ["Security", "Cyber"],
    summary:
      "A recent breach shows how employee device misuse can create serious risk if internal policies are bypassed.",
    url: "https://example.com/3",
    coverageCount: 2,
    isDuplicateGroup: false,
  },
  {
    id: "a4",
    title: "Startup funding rebounds as investors return to AI products",
    source: "TechCrunch",
    publishedAt: "2026-02-13T16:05:00Z",
    tags: ["Startups", "AI"],
    summary:
      "Funding is showing signs of recovery, especially in AI-first products with clear revenue models.",
    url: "https://example.com/4",
    coverageCount: 3,
    isDuplicateGroup: false,
  },
  {
    id: "a5",
    title: "Major policy update impacts international travel rules",
    source: "Times of India",
    publishedAt: "2026-02-13T18:55:00Z",
    tags: ["World", "Policy"],
    summary:
      "New travel guidelines introduce updated requirements, affecting multiple routes and airlines.",
    url: "https://example.com/5",
    coverageCount: 1,
    isDuplicateGroup: false,
  },
];

function statusBadge(status) {
  if (status === "working") return <Badge className="rounded-xl bg-green-500/15 text-green-400 border border-green-500/20">
      Working
    </Badge>
  if (status === "slow") return <Badge className="rounded-xl bg-yellow-500/15 text-yellow-400 border border-yellow-500/20">
      Slow
    </Badge>
  return  <Badge className="rounded-xl bg-red-500/15 text-red-400 border border-red-500/20">
      Failed
    </Badge>
}

function PageShell({ title, subtitle, children, rightActions }) {
  return (
    <div className="w-full">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
          {subtitle ? (
            <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
          ) : null}
        </div>

        {rightActions ? <div className="flex items-center gap-2">{rightActions}</div> : null}
      </div>

      <Separator className="my-5" />
      {children}
    </div>
  );
}

function SidebarItem({ icon: Icon, label, active, onClick }) {
  const IconComponent = Icon;

  return (
    <button
      onClick={onClick}
     className={`flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm transition
  ${
    active
      ? "bg-slate-900 text-white shadow-sm dark:bg-white dark:text-black"
      : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-white/5"
  }`}

    >
      <IconComponent size={18} />
      <span className="font-medium">{label}</span>
    </button>
  );
}


export default function App() {
  const [activePage, setActivePage] = useState("dashboard");

  // date range (simple for UI now)
  const [fromDate, setFromDate] = useState(new Date());
  const [toDate, setToDate] = useState(new Date());

  const [selectedSourceIds, setSelectedSourceIds] = useState(() =>
    new Set(["reuters", "bbc", "techcrunch"])
  );

  const [search, setSearch] = useState("");
  const [sortOrder, setSortOrder] = useState("newest"); // newest | oldest
const [selectedCategories, setSelectedCategories] = useState(new Set());
const [selectedFilterSources, setSelectedFilterSources] = useState(new Set());
const [selectedTags, setSelectedTags] = useState(new Set());

const [onlyDuplicates, setOnlyDuplicates] = useState(false);
const [onlyUnique, setOnlyUnique] = useState(false);
const [onlyTopRanked, setOnlyTopRanked] = useState(false);
const [totalFetched, setTotalFetched] = useState(0);

const [filtersOpen, setFiltersOpen] = useState(false);
const CATEGORY_OPTIONS = ["Tech", "World", "Business", "Security", "Sports"];

const TAG_OPTIONS = [
  "AI",
  "Elections",
  "Cybersecurity",
  "Market",
  "Policy",
  "Startups",
  "Economy",
  "Big Tech",
];


  
  
  
  const [selectedTopIds, setSelectedTopIds] = useState(new Set());

  const [agentRunning, setAgentRunning] = useState(false);
  const [articles, setArticles] = useState([]);
const [executiveSummary, setExecutiveSummary] = useState("");

const [deepSummaries, setDeepSummaries] = useState({});
const [deepLoading, setDeepLoading] = useState({});
const [darkMode, setDarkMode] = useState(false);
const [sourceHealth, setSourceHealth] = useState({});
const [testingSource, setTestingSource] = useState(null);

useEffect(() => {
  const root = document.documentElement;
  if (darkMode) root.classList.add("dark");
  else root.classList.remove("dark");
}, [darkMode]);


  const selectedSources = useMemo(() => {
    return DUMMY_SOURCES.filter((s) => selectedSourceIds.has(s.id));
  }, [selectedSourceIds]);

const filteredArticles = useMemo(() => {
  let list = [...articles];

  // 1) Search filter
  const q = search.trim().toLowerCase();
  if (q) {
    list = list.filter((a) => {
      return (
        (a.title || "").toLowerCase().includes(q) ||
        (a.source || "").toLowerCase().includes(q) ||
        (a.category || "").toLowerCase().includes(q) ||
        (a.tags || []).join(" ").toLowerCase().includes(q)
      );
    });
  }

  // 2) Category filter
  if (selectedCategories.size > 0) {
    list = list.filter((a) => selectedCategories.has(a.category));
  }

  // 3) Source filter
  if (selectedFilterSources.size > 0) {
    list = list.filter((a) => selectedFilterSources.has(a.source));
  }

  // 4) Tags filter (article must contain ALL selected tags)
  if (selectedTags.size > 0) {
    list = list.filter((a) => {
      const tags = new Set(a.tags || []);
      for (const t of selectedTags) {
        if (!tags.has(t)) return false;
      }
      return true;
    });
  }

  // 5) Duplicate filters
  if (onlyDuplicates) {
    list = list.filter((a) => a.isDuplicateGroup);
  }
  if (onlyUnique) {
    list = list.filter((a) => !a.isDuplicateGroup);
  }

  // 6) Sort
  list.sort((a, b) => {
    const ta = new Date(a.publishedAt).getTime();
    const tb = new Date(b.publishedAt).getTime();
    return sortOrder === "newest" ? tb - ta : ta - tb;
  });

  // 7) Top ranked only
  if (onlyTopRanked) {
    list = list.slice(0, 10);
  }

  return list;
}, [
  articles,
  search,
  selectedCategories,
  selectedFilterSources,
  selectedTags,
  onlyDuplicates,
  onlyUnique,
  onlyTopRanked,
  sortOrder,
]);


  function toggleSource(id) {
    setSelectedSourceIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleTopStory(id) {
    setSelectedTopIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else {
        // top 5 constraint
        if (next.size >= 5) return next;
        next.add(id);
      }
      return next;
    });
  }

  function toggleSetValue(setter, value) {
  setter((prev) => {
    const copy = new Set(prev);
    if (copy.has(value)) copy.delete(value);
    else copy.add(value);
    return copy;
  });
}

function clearAllFilters() {
  setSelectedCategories(new Set());
  setSelectedFilterSources(new Set());
  setSelectedTags(new Set());
  setOnlyDuplicates(false);
  setOnlyUnique(false);
  setOnlyTopRanked(false);
  setSortOrder("newest");
}

async function generateDeepSummary(article) {
  const id = article.id;

  // mark loading
  setDeepLoading((prev) => ({ ...prev, [id]: true }));

  try {
   const res = await fetch("http://127.0.0.1:8000/agent/deep-summary-url", {

      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
   body: JSON.stringify({
  title: article.title,
  url: article.url,
}),

    });

    const data = await res.json();

    setDeepSummaries((prev) => ({
      ...prev,
      [id]: data.deep_summary || "No summary generated.",
    }));
  } catch (err) {
    console.error(err);
    setDeepSummaries((prev) => ({
      ...prev,
      [id]: "Backend error while generating summary.",
    }));
  } finally {
    setDeepLoading((prev) => ({ ...prev, [id]: false }));
  }
}

async function runAgentFake() {
  setAgentRunning(true);

  try {
    const res = await fetch("http://127.0.0.1:8000/agent/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        fromDate: fromDate.toISOString(),
        toDate: toDate.toISOString(),
        sources: Array.from(selectedSourceIds),
      }),
    });

    const data = await res.json();

console.log("Agent response:", data);

setExecutiveSummary(data.executive_summary || "");
setArticles(data.articles || []);
setTotalFetched(data.total_fetched || 0);;
setSelectedTopIds(new Set()); // reset top 5 selection
setSearch(""); // reset search
 setDeepSummaries({});
    setDeepLoading({});
setAgentRunning(false);
setActivePage("stories");

  } catch (err) {
    console.error(err);
    setAgentRunning(false);
    alert("Backend error. Is FastAPI running on port 8000?");
  }
}

// export working

// ✅ PASTE THIS RIGHT HERE
async function exportStories() {
  const selected = articles.filter(a =>
    selectedTopIds.has(a.id)
  );

  const res = await fetch("http://127.0.0.1:8000/agent/export/email", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      executive_summary: executiveSummary,
      articles: selected
    })
  });

  const data = await res.json();

  const newWindow = window.open();
  newWindow.document.write(data.html);
}
// csv 
async function exportHistory() {
  const res = await fetch("http://127.0.0.1:8000/agent/export/csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      articles: articles
    })
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "history_export.csv";
  a.click();
}

// test source health 
async function testSource(sourceName) {
  setTestingSource(sourceName);

  try {
    const res = await fetch("http://127.0.0.1:8000/agent/test-source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_name: sourceName,
      }),
    });

    const data = await res.json();

    setSourceHealth(prev => ({
      ...prev,
      [sourceName]: {
        ...data,
        last_checked: new Date().toLocaleTimeString()
      }
    }));

  } catch (err) {
    setSourceHealth(prev => ({
      ...prev,
      [sourceName]: { status: "failed" }
    }));
  }

  setTestingSource(null);
}


function SidebarContent() {
  return (
    <div className="rounded-2xl border bg-card
 p-4 shadow-sm">
      <div className="flex items-center gap-2">
        <div className="grid h-9 w-9 place-items-center rounded-xl bg-slate-900 text-white">
          <Sparkles size={18} />
        </div>
        <div>
          <p className="text-sm font-semibold leading-none">
            News Intelligence
          </p>
          <p className="text-xs text-muted-foreground">Agent Workspace</p>
        </div>
      </div>

      <Separator className="my-4" />

      <div className="space-y-1">
        <SidebarItem
          icon={LayoutDashboard}
          label="Dashboard"
          active={activePage === "dashboard"}
          onClick={() => setActivePage("dashboard")}
        />
        <SidebarItem
          icon={Newspaper}
          label="Stories"
          active={activePage === "stories"}
          onClick={() => setActivePage("stories")}
        />
        {/* <SidebarItem
          icon={Mail}
          label="Email Preview"
          active={activePage === "email"}
          onClick={() => setActivePage("email")}
        /> */}
        <SidebarItem
          icon={Globe}
          label="Sources"
          active={activePage === "sources"}
          onClick={() => setActivePage("sources")}
        />
        <SidebarItem
          icon={History}
          label="History"
          active={activePage === "history"}
          onClick={() => setActivePage("history")}
        />
        <SidebarItem
  icon={MessageCircle}
  label="Ask AI"
  active={activePage === "ask"}
  onClick={() => setActivePage("ask")}
/>
      </div>

      <Separator className="my-4" />

      <div className="text-xs text-muted-foreground">
        <p className="font-medium text-muted-foreground">Selected Sources</p>
        <p className="mt-1">
          {selectedSources.length} enabled • {DUMMY_SOURCES.length} total
        </p>
      </div>
    </div>
  );
}

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b border-border bg-white/70 backdrop-blur-md dark:bg-black/30">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-3">
        {/* Left */}
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-slate-900 text-white dark:bg-white dark:text-black">
            <Sparkles size={18} />
          </div>

          <div>
            <p className="text-sm font-semibold leading-none">
              News Intelligence Agent
            </p>
            <p className="text-xs text-muted-foreground">
              Executive News Workspace
            </p>
          </div>
        </div>

        {/* Right */}
        <div className="flex items-center gap-2">
  {/* Date Range */}
  <div className="rounded-xl border border-border bg-background/60 px-4 py-2 text-sm text-foreground shadow-sm backdrop-blur-md">
    {formatDate(fromDate)} – {formatDate(toDate)}
  </div>

  {/* Theme Toggle */}
  <Button
    variant="outline"
    className="h-10 rounded-xl border-border bg-background/60 px-4 text-foreground shadow-sm backdrop-blur-md hover:bg-accent"
    onClick={() => setDarkMode((p) => !p)}
  >
    {darkMode ? (
      <Sun size={16} className="mr-2" />
    ) : (
      <Moon size={16} className="mr-2" />
    )}
    {darkMode ? "Light" : "Dark"}
  </Button>

  {/* Export */}
  <Button
    variant="outline"
    className="h-10 rounded-xl border-border bg-background/60 px-4 text-foreground shadow-sm backdrop-blur-md hover:bg-accent"
     onClick={exportStories}
  >
    Export
  </Button>
</div>

        
      </div>
    </header>
    <div className="mx-auto flex max-w-[1400px] gap-6 px-4 sm:px-6 py-6 items-start">

        {/* Sidebar */}
    <aside className="hidden w-[260px] shrink-0 lg:block sticky top-24 self-start h-[calc(100vh-6rem)] overflow-y-auto">

  <SidebarContent />
</aside>


        {/* Main */}
        <main className="flex-1">
          <div className="mb-4 flex items-center justify-between lg:hidden">
  <Sheet>
    <SheetTrigger asChild>
      <Button variant="outline" className="rounded-xl">
        <Menu size={18} className="mr-2" />
        Menu
      </Button>
    </SheetTrigger>
<SheetContent side="left" className="w-[320px] p-4">
  <SidebarContent />
</SheetContent>

  </Sheet>

  <p className="text-sm font-semibold text-muted-foreground">News Intelligence</p>
</div>

          {activePage === "dashboard" ? (
            <PageShell
              title="Dashboard"
              subtitle="Select date range and sources, then run the agent to generate your digest."
             

            >
              <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
                {/* Controls */}
               <div className="lg:col-span-1 lg:sticky lg:top-24 h-fit">
                 <Card className="lg:col-span-1 rounded-2xl">
                  <CardHeader>
                    <CardTitle className="text-base">Run Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div>
                      <p className="text-sm font-medium">Date Range</p>
                      <div className="mt-2 grid grid-cols-2 gap-3">
                        <Popover>
                          <PopoverTrigger asChild>
                            <Button variant="outline" className="justify-start">
                              {format(fromDate, "dd MMM yyyy")}
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-auto p-2">
                            <Calendar
                              mode="single"
                              selected={fromDate}
                              onSelect={(d) => d && setFromDate(d)}
                            />
                          </PopoverContent>
                        </Popover>

                        <Popover>
                          <PopoverTrigger asChild>
                            <Button variant="outline" className="justify-start">
                              {format(toDate, "dd MMM yyyy")}
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-auto p-2">
                            <Calendar
                              mode="single"
                              selected={toDate}
                              onSelect={(d) => d && setToDate(d)}
                            />
                          </PopoverContent>
                        </Popover>
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <p className="text-sm font-medium">Sources</p>
                     <div className="mt-3 max-h-[320px] overflow-y-auto space-y-3 pr-2 thin-scrollbar">
                        {DUMMY_SOURCES.map((s) => (
                          <div
                            key={s.id}
                            className="flex items-center justify-between gap-3 rounded-xl border px-3 py-2"
                          >
                            <div className="min-w-0">
                              <p className="truncate text-sm font-medium">{s.name}</p>
                              <p className="text-xs text-muted-foreground">
                                {s.type}
                              </p>
                            </div>

                            <div className="flex items-center gap-2">
                              {statusBadge(sourceHealth[s.name]?.status || s.status)}
                              <Checkbox
                                checked={selectedSourceIds.has(s.id)}
                                onCheckedChange={() => toggleSource(s.id)}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
<div className="pt-2">
  
                   <Button
  className="w-full rounded-xl bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-black dark:hover:bg-slate-200 shadow-md"
  onClick={runAgentFake}
  disabled={agentRunning || selectedSourceIds.size === 0}
>

                      {agentRunning ? "Running Agent..." : "Run Agent"}
                      <ChevronRight className="ml-2" size={18} />
                    </Button>
</div>

                    {agentRunning ? (
                      <div className="rounded-xl border bg-slate-50 p-3 text-sm text-muted-foreground">
                        <p className="font-medium">Agent pipeline</p>
                        <ul className="mt-2 list-disc pl-5 text-xs text-muted-foreground">
                          <li>Collecting sources…</li>
                          <li>Extracting articles…</li>
                          <li>Deduplicating…</li>
                          <li>Summarizing with OpenAI…</li>
                          <li>Ranking top stories…</li>
                        </ul>
                      </div>
                    ) : null}
                  </CardContent>
                </Card>




               </div>
                {/* Executive summary */}
                <Card className="lg:col-span-2 rounded-2xl ">
                  <CardHeader>
                    <CardTitle className="text-lg font-bold tracking-tight">Executive Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                  <div className="mt-2 space-y-3">

  {executiveSummary
    ?.split("\n")
    .filter(line => line.trim() !== "")
    .map((line, i) => (
      <div
        key={i}
        className="flex items-start gap-3 rounded-lg bg-muted/30 px-3 py-2 border border-border/40 hover:bg-muted/40 transition-colors"

      >
        <div className="mt-[6px] h-2 w-2 rounded-full bg-primary"></div>

        <p className="text-sm leading-6 text-foreground font-medium">
          {line.replace(/^-/, "").trim()}
        </p>

      </div>
  ))}

</div>

                    <Separator className="my-4" />

                    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                      <Card className="rounded-2xl glass-panel bg-card/70 backdrop-blur-md border border-white/10 dark:border-white/5">

                        <CardContent className="p-4">
                          <p className="text-xs text-muted-foreground">Articles Fetched</p>
                          <p className="mt-1 text-2xl font-semibold">{totalFetched}</p>

                        </CardContent>
                      </Card>

                      <Card className="rounded-2xl glass-panel bg-card/70 backdrop-blur-md border border-white/10 dark:border-white/5">

                        <CardContent className="p-4">
                          <p className="text-xs text-muted-foreground">Duplicates Grouped</p>
                         <p className="mt-1 text-2xl font-semibold">
  {articles.filter((a) => a.isDuplicateGroup).length}
</p>

                        </CardContent>
                      </Card>

                      <Card className="rounded-2xl glass-panel bg-card/70 backdrop-blur-md border border-white/10 dark:border-white/5">

                        <CardContent className="p-4">
                          <p className="text-xs text-muted-foreground">Sources Used</p>
                          <p className="mt-1 text-2xl font-semibold">
                            {selectedSources.length}
                          </p>
                        </CardContent>
                      </Card>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </PageShell>
          ) : null}

{activePage === "stories" ? (
  <PageShell
    title="Stories"
    subtitle="Browse stories, group duplicates, and select your top 5 for email delivery."
  >

    {/* ========================= */}
    {/* 🔥 STICKY SECOND HEADER */}
    {/* ========================= */}
    <div className="sticky top-16 z-40 bg-background border-b border-border pb-4 pt-2">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">

        {/* LEFT SIDE */}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">

          {/* Search */}
          <div className="relative">
            <Search
              className="absolute left-3 top-2.5 text-muted-foreground"
              size={16}
            />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search stories..."
              className="pl-9 w-full sm:w-[260px] rounded-xl"
            />
          </div>

          {/* Filters */}
          <Sheet open={filtersOpen} onOpenChange={setFiltersOpen}>
            <SheetTrigger asChild>
              <Button
                variant="outline"
                className="rounded-xl border-white/20 bg-white/60 backdrop-blur-md hover:bg-white/80 dark:bg-white/5 dark:hover:bg-white/10 dark:border-white/10"
              >
                <Filter size={16} className="mr-2" />
                Filters
              </Button>
            </SheetTrigger>

            <SheetContent side="right" className="w-[380px] p-0">
              <div className="h-full p-5">
                <div className="glass-panel rounded-2xl p-5 h-full overflow-auto thin-scrollbar">

                  <div className="flex items-center justify-between">
                    <p className="text-base font-semibold">Filters</p>
                    <Button
                      variant="ghost"
                      className="rounded-xl"
                      onClick={clearAllFilters}
                    >
                      Clear all
                    </Button>
                  </div>

                  <Separator className="my-4" />

                  {/* Sort */}
                  <div>
                    <p className="text-sm font-medium">Sort by time</p>
                    <div className="mt-2 flex gap-2">
                      <Button
                        variant={sortOrder === "newest" ? "default" : "outline"}
                        className="rounded-xl"
                        onClick={() => setSortOrder("newest")}
                      >
                        Newest
                      </Button>
                      <Button
                        variant={sortOrder === "oldest" ? "default" : "outline"}
                        className="rounded-xl"
                        onClick={() => setSortOrder("oldest")}
                      >
                        Oldest
                      </Button>
                    </div>
                  </div>

                  <Separator className="my-4" />

                  {/* Category */}
                  <div>
                    <p className="text-sm font-medium">Category</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {CATEGORY_OPTIONS.map((c) => {
                        const active = selectedCategories.has(c);
                        return (
                          <Button
                            key={c}
                            variant={active ? "default" : "outline"}
                            className="rounded-xl"
                            onClick={() =>
                              toggleSetValue(setSelectedCategories, c)
                            }
                          >
                            {c}
                          </Button>
                        );
                      })}
                    </div>
                  </div>

                  <Separator className="my-4" />

                  {/* Sources */}
                  <div>
                    <p className="text-sm font-medium">Sources</p>
                    <div className="mt-3 space-y-2">
                      {selectedSources.map((s) => (
                        <div
                          key={s.id}
                          className="flex items-center justify-between rounded-xl border border-white/10 px-3 py-2"
                        >
                          <p className="text-sm">{s.name}</p>
                          <Checkbox
                            checked={selectedFilterSources.has(s.name)}
                            onCheckedChange={() =>
                              toggleSetValue(setSelectedFilterSources, s.name)
                            }
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <Separator className="my-4" />

                  {/* Duplicate */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">Show only duplicates</p>
                      <Switch
                        checked={onlyDuplicates}
                        onCheckedChange={setOnlyDuplicates}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">Show only unique</p>
                      <Switch
                        checked={onlyUnique}
                        onCheckedChange={setOnlyUnique}
                      />
                    </div>
                  </div>

                  <Separator className="my-4" />

                  {/* Tags */}
                  {/* <div>
                    <p className="text-sm font-medium">Tags</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Select tags to narrow down stories.
                    </p>

                    <div className="mt-3 flex flex-wrap gap-2">
                      {TAG_OPTIONS.map((t) => {
                        const active = selectedTags.has(t);
                        return (
                          <Button
                            key={t}
                            variant={active ? "default" : "outline"}
                            className="rounded-xl"
                            onClick={() =>
                              toggleSetValue(setSelectedTags, t)
                            }
                          >
                            {t}
                          </Button>
                        );
                      })}
                    </div>
                  </div> */}

                  <Separator className="my-4" />

                  {/* Top Ranked */}
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">Only Top Ranked</p>
                      <p className="text-xs text-muted-foreground">
                        Show top 10 stories only
                      </p>
                    </div>
                    <Switch
                      checked={onlyTopRanked}
                      onCheckedChange={setOnlyTopRanked}
                    />
                  </div>

                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>

        {/* Email Button */}
        {/* <Button
          className="rounded-xl"
          onClick={() => setActivePage("email")}
          disabled={selectedTopIds.size === 0}
        >
          Email Preview
          <ChevronRight className="ml-2" size={18} />
        </Button> */}
      </div>
    </div>

    {/* ========================= */}
    {/* 🔥 SCROLLABLE STORIES */}
    {/* ========================= */}
    <div className="max-h-[calc(100vh-220px)] overflow-y-auto thin-scrollbar pr-2">
      <div className="grid grid-cols-1 gap-4 pt-4">

        {filteredArticles.map((a) => {
          const selected = selectedTopIds.has(a.id);

          return (
            <Card key={a.id} className="rounded-2xl">
              <CardContent className="p-5">
                <div className="flex items-start justify-between gap-4">

                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="truncate text-base font-semibold">
                        {a.title}
                      </h3>

                      {a.isDuplicateGroup && (
                        <Badge variant="secondary">
                          Covered by {a.coverageCount} sources
                        </Badge>
                      )}
                    </div>

                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
  <span>{a.source}</span>

  <span>•</span>
  <span >
    {a.reading_time || "1 min read"}
  </span>

  <span>•</span>
  <span>{new Date(a.publishedAt).toLocaleString()}</span>
</div>

                    <p className="mt-3 text-sm text-muted-foreground">
                      {a.summary}
                    </p>

                    {/* 🔥 Deep Summary (Working Like Original) */}
                    <div className="mt-4">
                      <Button
                        variant="outline"
                        className="rounded-xl border-border bg-background text-foreground hover:bg-accent"
                        onClick={() => generateDeepSummary(a)}
                        disabled={deepLoading[a.id]}
                      >
                        {deepLoading[a.id] ? "Generating..." : "Deep Summary"}
                      </Button>

                      {deepSummaries[a.id] && (
                       <div className="mt-4 rounded-2xl border border-border bg-card/60 p-6 shadow-md backdrop-blur-md transition-all">
                          <div className="mb-3 flex items-center justify-between">
                            <p className="text-sm font-bold tracking-wide text-primary">
                              Executive Brief
                            </p>
                            <span className="text-xs text-slate-400 dark:text-slate-500">
                              OpenAI
                            </span>
                          </div>

                        <div className="space-y-4">

{deepSummaries[a.id]
  .split("\n")
  .filter(line => line.trim() !== "")
  .map((line, i) => {

    const cleanLine = line.replace(/\*\*/g, "").trim().toLowerCase()

    const isHeading =
      cleanLine.startsWith("tl;dr") ||
      cleanLine.startsWith("what happened") ||
      cleanLine.startsWith("why it matters") ||
      cleanLine.startsWith("what’s next") ||
      cleanLine.startsWith("what's next")

    if (isHeading) {
      return (
        <h4
          key={i}
          className="text-sm font-semibold text-primary border-l-4 border-primary pl-3 mt-2"
        >
          {line.replace(/\*\*/g, "")}
        </h4>
      )
    }

    return (
      <p
        key={i}
        className="text-sm leading-6 text-muted-foreground pl-4"
      >
        {line}
      </p>
    )
  })}

</div>
                        </div>
                      )}
                    </div>

                    {/* Tags */}
                    <div className="mt-3 flex flex-wrap gap-2">
                      {a.tags.map((t) => (
                        <Badge key={t} variant="outline" className="rounded-xl">
                          {t}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Right Side */}
                  <div className="flex shrink-0 flex-col items-end gap-2">
                    <Button
                      variant={selected ? "default" : "outline"}
                      className="rounded-xl"
                      onClick={() => toggleTopStory(a.id)}
                    >
                      {selected ? "Selected" : "Add to Top 5"}
                    </Button>

                    <Button
                      variant="ghost"
                      className="rounded-xl"
                      onClick={() => window.open(a.url, "_blank")}
                    >
                      <ExternalLink size={16} className="mr-2" />
                      Open
                    </Button>

                    <p className="text-xs text-muted-foreground">
                      Selected: {selectedTopIds.size}/5
                    </p>
                  </div>

                </div>
              </CardContent>
            </Card>
          );
        })}

      </div>
    </div>

  </PageShell>
) : null}

          {/* {activePage === "email" ? (
            <PageShell
              title="Email Preview"
              subtitle="Preview the top 5 stories and send them with summaries."
              rightActions={
                <Button className="rounded-xl">
                  <Send size={16} className="mr-2" />
                  Send Email
                </Button>
              }
            >
              <Card className="rounded-2xl glass-panel bg-card/70 backdrop-blur-md border border-white/10 dark:border-white/5">

                <CardHeader>
                  <CardTitle className="text-base">Digest Email</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="rounded-xl border bg-slate-50 p-4">
                    <p className="text-sm font-medium">Subject</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      News Digest • {format(fromDate, "dd MMM")} –{" "}
                      {format(toDate, "dd MMM yyyy")}
                    </p>
                  </div>

                  <div className="rounded-xl border p-4">
                    <p className="text-sm font-medium">Recipient</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      your.email@company.com
                    </p>
                  </div>

                  <Separator />

                  <div className="space-y-3">
                    {articles.filter((a) => selectedTopIds.has(a.id)).map(
                      (a, idx) => (
                        <div
                          key={a.id}
                          className="rounded-xl border p-4"
                        >
                          <p className="text-xs text-muted-foreground">
                            Story #{idx + 1} • {a.source}
                          </p>
                          <p className="mt-1 text-sm font-semibold">{a.title}</p>
                          <p className="mt-2 text-sm text-muted-foreground">
                            {a.summary}
                          </p>
                          <Button
                            variant="link"
                            className="px-0"
                            onClick={() => window.open(a.url, "_blank")}
                          >
                            Read full story
                          </Button>
                        </div>
                      )
                    )}
                  </div>
                </CardContent>
              </Card>
            </PageShell>
          ) : null} */}

          {activePage === "sources" ? (
            <PageShell
              title="Sources"
              subtitle="Manage your sources, types (RSS/NewsAPI), and monitor health."
            >
             <Card className="rounded-2xl glass-panel bg-card/70 backdrop-blur-md border border-white/10 dark:border-white/5">

                <CardHeader>
                  <CardTitle className="text-base">Configured Sources</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {DUMMY_SOURCES.map((s) => (
                   <div
  key={s.id}
  className="rounded-xl border px-4 py-3 space-y-3"
>
  {/* Top Row */}
  <div className="flex items-center justify-between gap-3">
    <div>
      <p className="text-sm font-semibold">{s.name}</p>
      <p className="text-xs text-muted-foreground">
        {s.type}
      </p>
    </div>

    <div className="flex items-center gap-3">
     {statusBadge(sourceHealth[s.name]?.status || s.status)}

      <Button
        variant="outline"
        className="rounded-xl"
        onClick={() => testSource(s.name)}
      >
        {testingSource === s.name ? "Testing..." : "Test"}
      </Button>
    </div>
  </div>

  {/* 🔥 HEALTH PANEL */}
  {sourceHealth[s.name] && (
    <div className="mt-2 text-xs space-y-1 text-muted-foreground rounded-lg bg-background/50 p-3 border">

      <div>
        Status:{" "}
       {sourceHealth[s.name].status === "working" && "🟢 Working"}
        {sourceHealth[s.name].status === "slow" && "🟡 Slow"}
        {sourceHealth[s.name].status === "failed" && "🔴 Failed"}
      </div>

      {sourceHealth[s.name].latency_ms && (
        <div>Latency: {sourceHealth[s.name].latency_ms} ms</div>
      )}

      {sourceHealth[s.name].articles_found && (
        <div>Articles: {sourceHealth[s.name].articles_found}</div>
      )}

      {sourceHealth[s.name].latest_article_date && (
        <div>Latest: {sourceHealth[s.name].latest_article_date}</div>
      )}

      {sourceHealth[s.name].preview_titles?.length > 0 && (
        <div>
          Top Headlines:
          <ul className="list-disc ml-4 mt-1 space-y-1">
            {sourceHealth[s.name].preview_titles.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="opacity-60 text-[11px]">
        Last Checked: {sourceHealth[s.name].last_checked}
      </div>

    </div>
  )}

</div>
                  ))}
                </CardContent>
              </Card>
            </PageShell>
          ) : null}

          {activePage === "history" ? (
            <PageShell
              title="History"
              subtitle="View previous agent runs and open results instantly."
            >
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {[1, 2, 3, 4].map((x) => (
                  <Card key={x} className="rounded-2xl">
                    <CardContent className="p-5">
                      <p className="text-xs text-muted-foreground">
                        Run #{x}
                      </p>
                      <p className="mt-1 text-sm font-semibold">
                        {format(new Date(), "dd MMM yyyy")} • 5 sources
                      </p>
                      <p className="mt-2 text-sm text-muted-foreground">
                        38 articles fetched • 7 duplicates grouped • 5 selected
                      </p>
                      <div className="mt-4 flex gap-2">
                        <Button
                          className="rounded-xl"
                          onClick={() => setActivePage("stories")}
                        >
                          Open Results
                        </Button>
                        <Button variant="outline" className="rounded-xl"
                          onClick={exportHistory}
                        >
                          Export
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </PageShell>
          ) : null}

          {activePage === "ask" ? (
  <PageShell
    title="Ask AI"
    subtitle="Ask questions about today's news and get instant insights."
  >
    <AskAIPage articles={filteredArticles} />
  </PageShell>
) : null}
        </main>
      </div>
    </div>
  );
}
