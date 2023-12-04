export interface HistoryProps {
  date: string;
  queries: Array<{
    query: string;
    time: string;
  }>;
}
