import { ReactNode } from "react";

export interface ButtonProps {
  className: string;
  rightIcon?: any;
  leftIcon?: any;
  children?: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  isLoading?: boolean;
}
