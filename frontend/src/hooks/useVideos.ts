import * as React from "react";

const API_BASE = "http://localhost:8000";

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

export function useVideos(status: string = "ready", limit: number = 200) {
  const [videos, setVideos] = React.useState<VideoItem[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const reload = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/videos?status=${encodeURIComponent(status)}&limit=${limit}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setVideos(data);
    } catch (e: any) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }, [status, limit]);

  React.useEffect(() => { reload(); }, [reload]);

  return { videos, loading, error, reload, setVideos };
}
