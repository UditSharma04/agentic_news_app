import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const API_URL = "https://news-agent-fullstack-su-new.onrender.com";

export default function AskAIPage({ articles }) {
  const [question, setQuestion] = useState("");
  const [insight, setInsight] = useState("");
  const [matchedIds, setMatchedIds] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);

    const res = await fetch(`${API_URL}/agent/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        articles
      })
    });

    const data = await res.json();

    setInsight(data.insight);
    setMatchedIds(data.matched_ids);
    setLoading(false);
  };

  const matchedArticles = articles.filter(a =>
    matchedIds.includes(String(a.id))
  );

  return (
    <div className="space-y-6">

      <div className="flex gap-2">
        <input
          className="flex-1 rounded-xl border px-4 py-2 bg-background"
          placeholder="Ask about today's news..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <Button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking..." : "Ask"}
        </Button>
      </div>

      {insight && (
        <Card className="rounded-2xl">
          <CardContent className="p-5">
            <h3 className="font-semibold mb-2">🧠 AI Insight</h3>
            <p className="text-sm text-muted-foreground">{insight}</p>
          </CardContent>
        </Card>
      )}

      {matchedArticles.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-semibold">📰 Related Stories</h3>
          {matchedArticles.map(a => (
            <Card key={a.id} className="rounded-2xl">
              <CardContent className="p-5">
                <h4 className="font-semibold">{a.title}</h4>
                <p className="text-sm text-muted-foreground mt-2">
                  {a.summary}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

    </div>
  );
}