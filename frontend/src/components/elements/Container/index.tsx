import React from "react";
import { ContainerProps } from "./interface";

export const Container: React.FC<ContainerProps> = ({
  className,
  children,
  onClick,
  isLoading,
}) => {
  return (
    <>
      <div
        onClick={onClick}
        className={`px-6 py-4 bg-primaryContainer rounded-xl flex-row justify-start items-center gap-4 inline-flex ${className} ${
          isLoading
            ? "border-black disabled:bg-orange-dark"
            : "disabled:bg-cream-normal"
        } disabled:text-primaryContainer disabled:drop-shadow-none disabled:bg-transparent`}
      >
        {isLoading ? (
          <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-inherit"></div>
        ) : (
          children
        )}
      </div>
    </>
  );
};
