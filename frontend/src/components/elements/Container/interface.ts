import { ReactNode } from "react";

export interface ContainerProps {
  className: string;
  children?: ReactNode;
  onClick?: () => void;
  isLoading?: boolean;
}
