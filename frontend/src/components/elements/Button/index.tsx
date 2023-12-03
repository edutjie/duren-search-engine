import React from "react";
import { ButtonProps } from "./interface";

export const Button: React.FC<ButtonProps> = ({
  className,
  children,
  onClick,
  leftIcon,
  rightIcon,
  disabled,
  isLoading,
}) => {
  return (
    <>
      <button
        onClick={onClick}
        disabled={disabled || isLoading}
        className={` flex h-fit select-none items-center justify-center gap-1 rounded-3xl text-sm font-medium transition-all 
          text-primaryContainer hover:drop-shadow-lg active:bg-cream-normal active:text-primary active:drop-shadow-none ${className} ${
          isLoading
            ? "border-black disabled:bg-orange-dark"
            : "disabled:bg-cream-normal"
        } disabled:text-primaryContainer disabled:drop-shadow-none disabled:bg-transparent`}
      >
        {leftIcon && <span className="stroke-current">{leftIcon}</span>}
        {isLoading ? (
          <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-inherit"></div>
        ) : (
          children
        )}
        {rightIcon && <span className="stroke-current">{rightIcon}</span>}
      </button>
    </>
  );
};
