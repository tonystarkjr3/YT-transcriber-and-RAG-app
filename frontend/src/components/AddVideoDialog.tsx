import * as React from "react";
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, Button, Stack, FormControlLabel, Checkbox, CircularProgress
} from "@mui/material";

const API_BASE = "http://localhost:8000";

export default function AddVideoDialog({
  open, onClose, onAdded,
}: { open: boolean; onClose: () => void; onAdded: () => void }) {
  const [url, setUrl] = React.useState("");
  const [lang, setLang] = React.useState("en");
  const [adv, setAdv] = React.useState(false);
  const [windowSize, setWindowSize] = React.useState(8);
  const [overlap, setOverlap] = React.useState(3);
  const [loading, setLoading] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!open) { setUrl(""); setErr(null); setAdv(false); setWindowSize(8); setOverlap(3); }
  }, [open]);

  async function submit() {
    if (!url.trim()) return;
    setLoading(true); setErr(null);
    try {
      const res = await fetch(`${API_BASE}/api/videos/add`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ urlOrId: url.trim(), lang, window_size: windowSize, overlap }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.status?.startsWith("failed")) throw new Error(data.status);
      onAdded(); // tell parent to refresh list
      onClose();
    } catch (e:any) {
      setErr(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Add YouTube Video</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{mt:1}}>
          <TextField
            label="YouTube URL or ID"
            value={url}
            onChange={(e)=>setUrl(e.target.value)}
            autoFocus
            fullWidth
          />
          <TextField
            label="Language"
            value={lang}
            onChange={(e)=>setLang(e.target.value)}
            fullWidth
            helperText="Transcript language code (e.g., en)"
          />
          <FormControlLabel
            control={<Checkbox checked={adv} onChange={(e)=>setAdv(e.target.checked)} />}
            label="Advanced options"
          />
          {adv && (
            <Stack direction="row" spacing={2}>
              <TextField
                label="Window size"
                type="number"
                value={windowSize}
                onChange={(e)=>setWindowSize(parseInt(e.target.value||"8",10))}
                inputProps={{ min: 2, max: 20 }}
              />
              <TextField
                label="Overlap"
                type="number"
                value={overlap}
                onChange={(e)=>setOverlap(parseInt(e.target.value||"3",10))}
                inputProps={{ min: 0, max: 10 }}
              />
            </Stack>
          )}
          {err && <div style={{color:"#c00"}}>{err}</div>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>Cancel</Button>
        <Button variant="contained" onClick={submit} disabled={loading}>
          {loading ? <CircularProgress size={18}/> : "Add"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
