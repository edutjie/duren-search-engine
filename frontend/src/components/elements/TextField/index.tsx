import Image from "next/image";
import React from "react";
import { MdOutlineSearch } from "react-icons/md";
import { TextFieldProps } from "./interface";

export const TextField: React.FC<TextFieldProps> = ({
  className,
  onChange,
  disabled,
  isLoading,
  placeholder,
  value,
}) => {
  return (
    <>
      <div
        className={`flex flex-row items-center gap-2 justify-between ${className}`}
      >
        <div className="flex flex-row gap-3 w-full">
          <div className="flex flex-row gap-1">
            <span className="text-icons text-3xl">
              <MdOutlineSearch />
            </span>
            <svg
              width="2"
              height="30"
              viewBox="0 0 2 30"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <line
                x1="0.5"
                y1="30"
                x2="0.5"
                y2="0.5"
                stroke="#CBE04C"
                strokeWidth="2"
              />
            </svg>
          </div>
          <input
            type="text"
            className={`w-full flex text-xl font-base transition-all outline-none text-primaryText placeholder-secondaryText  ${
              isLoading ? "cursor-not-allowed" : "cursor-text"
            }
          ${
            isLoading
              ? "border-black bg-orange-dark"
              : "border-cream-normal bg-cream-normal"
          }
          `}
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            disabled={disabled || isLoading}
          />
        </div>

        {isLoading && (
          <div className="ml-2">
            <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-inherit"></div>
          </div>
        )}
      </div>
    </>
  );
};
