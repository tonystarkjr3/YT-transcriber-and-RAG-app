import * as React from "react";
import {
  AppBar, Box, Toolbar, Typography, IconButton,
  Container, TextField, Stack, Tooltip, Button,
  InputAdornment, CircularProgress
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import ManageSearchIcon from "@mui/icons-material/ManageSearch";
import AddIcon from "@mui/icons-material/Add";

import TraceSidepanel from "../components/TraceSidepanel";
import AnswerCard from "../components/AnswerCard";
import AddVideoDialog from "../components/AddVideoDialog";
import VideosSelect, { VideoItem } from "../components/VideosSelect";
import { useVideos } from "../hooks/useVideos"

const API_BASE = "http://localhost:8000";

export default function Home() {
  const [adminOpen, setAdminOpen] = React.useState(false);
  const [adminSelectTrace, setAdminSelectTrace] = React.useState<string | null>(null);
  const [adminAuthed, setAdminAuthed] = React.useState<boolean>(false);

  const [addOpen, setAddOpen] = React.useState(false);

  const [query, setQuery] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [answer, setAnswer] = React.useState<string>("");
  const [sources, setSources] = React.useState<any[]>([]);
  const [scope, setScope] = React.useState<VideoItem[]>([]);
  const { videos, loading: videosLoading, reload: reloadVideos } = useVideos('ready', 200);

  // inside component, with other useState hooks
  const clearResults = React.useCallback(() => {
    setAnswer("");
    setSources([]);
    setError(null);
    setAdminSelectTrace(null);
  }, []);

  async function runQuery(e?: React.FormEvent) {
    e?.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setAnswer("");
    setSources([]);

    try {
      const video_ids = scope.map((v) => v.id);
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 5, debug: true, video_ids }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setAnswer(data.answer || "");
      setSources(data.sources || []);

      const traceId = data.debug?.trace_id;
      if (traceId) {
        setAdminSelectTrace(traceId);
      }
    } catch (e: any) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <AppBar position="sticky" elevation={0} sx={{ bgcolor: "white", color: "black", borderBottom: "1px solid #eee" }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 600 }}>
            YouTube RAG
          </Typography>
          <Tooltip title="Inspect in admin mode">
            <IconButton onClick={() => setAdminOpen(true)}>
              <ManageSearchIcon />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 6 }}>
        <Stack spacing={3}>
          <Typography variant="h4" fontWeight={700} textAlign="center">
            Ask questions about indexed YouTube videos
          </Typography>

          <Button startIcon={<AddIcon />} onClick={() => setAddOpen(true)}>
            Add video
          </Button>

          <Box component="form" onSubmit={runQuery}>
            <Stack direction="row" spacing={1}>
              <TextField
                fullWidth
                placeholder="e.g., What did the host say about topic X?"
                value={query}
                onChange={(e) => { setQuery(e.target.value); clearResults(); } }
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon sx={{ opacity: 0.6 }} />
                    </InputAdornment>
                  ),
                }}
              />
              <Button variant="contained" onClick={runQuery} disabled={loading}>
                {loading ? <CircularProgress size={18} /> : "Ask"}
              </Button>
              <Button
                onClick={() => {
                  setQuery("");
                  setScope([]);
                  clearResults();
                }}
              >
                Clear
              </Button>
            </Stack>
          </Box>

          {/* Scope selection */}
          <VideosSelect value={scope} onChange={(v) => { setScope(v); clearResults(); }} videos={videos} loading={videosLoading} />

          {error && <Typography color="error">{error}</Typography>}
          {!!answer && <AnswerCard answer={answer} sources={sources} />}
        </Stack>
      </Container>

      <AddVideoDialog
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onAdded={() => {
          reloadVideos();
        }}
      />

      <TraceSidepanel
        open={adminOpen}
        onClose={() => { setAdminOpen(false); setAdminSelectTrace(null); }}
        adminAuthed={adminAuthed}
        setAdminAuthed={setAdminAuthed}
        // @ts-ignore optional prop supported in your implementation
        autoSelectId={adminSelectTrace}
      />
    </>
  );
}
