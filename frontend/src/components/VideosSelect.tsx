import * as React from "react";
import {
  Autocomplete, TextField, Chip, Tooltip, Box, Typography
} from "@mui/material";

export type VideoItem = {
  id: number;           // DB PK
  video_id: string;     // YouTube ID
  title?: string | null;
  channel?: string | null;
  status: string;
  url?: string | null;
  lang?: string | null;
  duration_sec?: number | null;
  last_ingested_at?: string | null;
};

export default function VideosSelect({
  value, onChange, videos, loading = false,
}: {
  value: VideoItem[];
  onChange: (v: VideoItem[]) => void;
  videos: VideoItem[];
  loading?: boolean;
}) {
  return (
    <Autocomplete
      multiple
      options={videos}
      value={value}
      onChange={(_, v) => onChange(v)}
      disableCloseOnSelect
      loading={loading}
      isOptionEqualToValue={(a, b) => a.id === b.id}
      groupBy={(o) => o.channel ?? "Unknown channel"}
      getOptionLabel={(o) => {
        const title = o.title?.trim() || o.video_id;
        const channel = o.channel?.trim();
        return channel ? `${title} — ${channel} (${o.video_id})` : `${title} (${o.video_id})`;
      }}
      renderOption={(props, opt) => {
        const title = opt.title?.trim() || opt.video_id;
        const channel = opt.channel?.trim();
        return (
          <li {...props} key={opt.id}>
            <Tooltip
              arrow
              title={
                <Box sx={{ p: 0.5 }}>
                  <Typography variant="caption">
                    ID: <b>{opt.video_id}</b><br />
                    URL: {opt.url || `https://youtu.be/${opt.video_id}`}<br />
                    Lang: {opt.lang || "—"}{opt.duration_sec ? ` • ${Math.round(opt.duration_sec / 60)} min` : ""}
                    {opt.last_ingested_at ? ` • indexed ${new Date(opt.last_ingested_at).toLocaleString()}` : ""}
                  </Typography>
                </Box>
              }
            >
              <Box sx={{ display: "flex", flexDirection: "column" }}>
                <Typography variant="body2">
                  {title} {channel ? ` — ${channel}` : ""}{" "}
                  <Typography component="span" variant="caption" sx={{ opacity: 0.6 }}>
                    ({opt.video_id})
                  </Typography>
                </Typography>
              </Box>
            </Tooltip>
          </li>
        );
      }}
      renderTags={(selected, getTagProps) =>
        selected.map((opt, i) => {
          const label = `${opt.title?.trim() || opt.video_id} (${opt.video_id})`;
          return <Chip {...getTagProps({ index: i })} key={opt.id} label={label} />;
        })
      }
      renderInput={(params) => (
        <TextField
          {...params}
          label="Indexed videos (optional scope)"
          placeholder="Select Videos Grouped by Channel"
        />
      )}
    />
  );
}
