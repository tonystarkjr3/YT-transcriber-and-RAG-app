// src/components/TraceSidepanel.tsx
import * as React from "react";
import {
  Drawer, Box, IconButton, Typography, Divider, Stack,
  List, ListItemButton, ListItemText, CircularProgress,
  TextField, Button, Link, Chip
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

const API_BASE = "http://localhost:8000";

type TraceListItem = {
  id: string;
  ts: string;
  query: string;
  top_k: number;
  confidence: number;
};

type TraceHit = {
  id: number;
  score: number;
  source?: string;
  url?: string;
  published_at?: string;
  snippet: string;
  // new optional fields from backend
  start_sec?: number | null;
  dur_sec?: number | null;
  video_pk?: number | null;
};

type TraceDetail = {
  id: string;
  ts: string;
  query: string;
  top_k: number;
  confidence: any; // or a stricter interface if you prefer
  prompt_preview?: string;
  answer_preview?: string;
  providers?: Record<string, any>;
  embed_dim?: number;
  hits: TraceHit[];
};

function withTimestamp(url?: string, startSec?: number | null) {
  if (!url || startSec == null) return url || "";
  const s = Math.max(0, Math.floor(startSec));
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}t=${s}s`;
}

function toTimecode(startSec?: number | null) {
  if (startSec == null) return "—";
  const s = Math.max(0, Math.floor(startSec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}


function useTraces(open: boolean) {
  const [data, setData] = React.useState<TraceListItem[] | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    if (!open) return;
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/admin/traces`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [open]);

  React.useEffect(() => { load(); }, [load]);
  return { data, loading, error, reload: load };
}

export default function TraceSidepanel
  (
    { open, onClose, adminAuthed, setAdminAuthed }: 
    { open: boolean; onClose: () => void, adminAuthed: boolean, setAdminAuthed: React.Dispatch<React.SetStateAction<boolean>> }
  ) {
  // const [authed, setAuthed] = React.useState(false);
  const [pass, setPass] = React.useState("");
  const { data, loading, error, reload } = useTraces(open && adminAuthed);
  const [selected, setSelected] = React.useState<TraceDetail | null>(null);
  const [detailLoading, setDetailLoading] = React.useState(false);
  const [detailError, setDetailError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!open) { setSelected(null); setPass(""); }
  }, [open]);

  async function loadDetail(id: string) {
    setDetailLoading(true); setDetailError(null);
    try {
      const res = await fetch(`${API_BASE}/api/admin/traces/${id}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSelected(await res.json());
    } catch (e: any) {
      setDetailError(String(e));
    } finally {
      setDetailLoading(false);
    }
  }

  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: 900, maxWidth: "100%" } }}>
      <Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 1 }}>
        <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 700 }}>Admin: Retrieval Traces</Typography>
        <IconButton onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      <Divider />

      {!adminAuthed ? (
        <Box sx={{ p: 2 }}>
          <Typography variant="body2" sx={{ mb: 1, color: "text.secondary" }}>
            Dev-only access. Enter admin password.
          </Typography>
          <Stack direction="row" spacing={1}>
            <TextField
              size="small"
              type="password"
              value={pass}
              onChange={(e)=>setPass(e.target.value)}
              placeholder="admin"
            />
            <Button variant="contained" onClick={()=>setAdminAuthed(pass === "letmein")}>Enter</Button>
          </Stack>
        </Box>
      ) : (
        <Box sx={{ display: "grid", gridTemplateColumns: "340px 1fr", minHeight: 0 }}>
          {/* Left: timeline list */}
          <Box sx={{ borderRight: "1px solid #eee", height: "calc(100vh - 64px)", overflowY: "auto" }}>
            <Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>Recent Queries</Typography>
              <Button size="small" onClick={reload}>Reload</Button>
            </Box>

            {loading && <Box sx={{ p: 2 }}><CircularProgress size={20} /></Box>}
            {error && <Box sx={{ p: 2, color: "error.main" }}>{error}</Box>}
            <List dense>
              {(data || []).map((t) => (
                <ListItemButton key={t.id} onClick={() => loadDetail(t.id)} selected={selected?.id === t.id}>
                  <ListItemText
                    primary={t.query}
                    secondary={`${new Date(t.ts).toLocaleString()}  •  k=${t.top_k}  •  conf=${t.confidence}`}
                    primaryTypographyProps={{ noWrap: true }}
                    secondaryTypographyProps={{ noWrap: true }}
                  />
                </ListItemButton>
              ))}
              {!loading && (data || []).length === 0 && (
                <Box sx={{ p: 2, color: "text.secondary" }}>
                  No traces yet. Make a query with <code>debug:true</code>.
                </Box>
              )}
            </List>
          </Box>

          {/* Right: detail */}
          <Box sx={{ p: 2, height: "calc(100vh - 64px)", overflowY: "auto" }}>
            {!selected && <Typography sx={{ color: "text.secondary" }}>Select a trace…</Typography>}
            {detailLoading && <CircularProgress size={20} />}
            {detailError && <Box sx={{ color: "error.main" }}>{detailError}</Box>}

            {selected && !detailLoading && (
              <Stack spacing={2}>
                <Box>
                  <Typography variant="h6" fontWeight={700}>{selected.query}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {new Date(selected.ts).toLocaleString()} • k={selected.top_k} • dim={selected.embed_dim}
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                    <Chip size="small" label={`conf: ${selected.confidence.confidence}`} />
                    <Chip size="small" label={`sim: ${selected.confidence.sim_avg ?? "-"}`} />
                    <Chip size="small" label={`recency: ${selected.confidence.recency ?? "-"}`} />
                    <Chip size="small" label={`diversity: ${selected.confidence.src_diversity ?? "-"}`} />
                    <Chip size="small" label={`embed: ${selected.providers?.embed}`} />
                    <Chip size="small" label={`llm: ${selected.providers?.llm}`} />
                  </Stack>
                </Box>

                <Box>
                  <Typography variant="subtitle1" fontWeight={700}>Answer</Typography>
                  <Box sx={{ p: 1.5, bgcolor: "#fafafa", border: "1px solid #eee", borderRadius: 1 }}>
                    <Typography whiteSpace="pre-wrap">{selected.answer_preview || "—"}</Typography>
                  </Box>
                </Box>

                <Box>
                  <Typography variant="subtitle1" fontWeight={700}>Prompt preview</Typography>
                  <Box sx={{ p: 1.5, bgcolor: "#fafafa", border: "1px solid #eee", borderRadius: 1 }}>
                    <Typography whiteSpace="pre-wrap" sx={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 13 }}>
                      {selected.prompt_preview || "—"}
                    </Typography>
                  </Box>
                </Box>

                <Box>
                  <Typography variant="subtitle1" fontWeight={700}>Top-k sources</Typography>
                  <Stack spacing={1.5}>
                    {selected.hits.map((h, i) => (
                      <Box key={i} sx={{ p: 1.5, border: "1px solid #eee", borderRadius: 1 }}>
                        <Typography variant="body2" sx={{ mb: 0.5 }}>
                          <b>#{i+1}</b> • score {h.score} • {h.source || "—"} • {toTimecode(h.start_sec)}
                        </Typography>

                        {h.url ? (
                          <Link href={withTimestamp(h.url, h.start_sec)} target="_blank" rel="noreferrer" underline="hover">
                            {new URL(h.url).hostname}
                          </Link>
                        ) : (
                          <Typography variant="body2" color="text.secondary">No URL</Typography>
                        )}

                        <Typography variant="body2" sx={{ mt: 0.5 }}>{h.snippet}</Typography>
                      </Box>
                    ))}
                    {selected.hits.length === 0 && <Typography color="text.secondary">No hits.</Typography>}
                  </Stack>
                </Box>
              </Stack>
            )}
          </Box>
        </Box>
      )}
    </Drawer>
  );
}
