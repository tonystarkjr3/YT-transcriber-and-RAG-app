import * as React from "react";
import { Card, CardContent, Typography, Link, Divider } from "@mui/material";

interface Source {
  source?: string;
  url?: string;
  score?: number;
  snippet?: string;
}

interface AnswerCardProps {
  answer: string;
  sources: Source[];
}

const AnswerCard: React.FC<AnswerCardProps> = ({ answer, sources }) => {
  return (
    <Card sx={{ mb: 2, boxShadow: 3, borderRadius: 2 }}>
      <CardContent>
        <Typography variant="body1" sx={{ mb: 2 }}>
          {answer}
        </Typography>

        {sources.length > 0 && (
          <>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle2" color="text.secondary">
              Sources
            </Typography>
            {sources.map((s, i) => (
              <Typography key={i} variant="body2" sx={{ mt: 0.5 }}>
                {s.url ? (
                  <Link href={s.url} target="_blank" rel="noopener">
                    {s.source || "link"}
                  </Link>
                ) : (
                  s.source
                )}
                {s.score !== undefined && ` (score: ${s.score.toFixed(2)})`}
                {s.snippet && ` — ${s.snippet}`}
              </Typography>
            ))}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default AnswerCard;