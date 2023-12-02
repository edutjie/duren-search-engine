import React, { useState } from "react";
import { DropdownProps } from "./interface";
import { MdArrowDropDown } from "react-icons/md";

export const Dropdown: React.FC<DropdownProps> = ({
  className,
  onChange,
  disabled,
  isLoading,
  placeholder,
  value,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    if (!disabled && !isLoading) {
      setIsOpen(!isOpen);
    }
  };

  const handleSelect = (selectedValue: string) => {
    if (onChange && !disabled && !isLoading) {
      setIsOpen(false);
      onChange(selectedValue);
      value = selectedValue;
    }
  };

  return (
    <div className={`${className}`}>
      <div>
        <button
          type="button"
          className={`flex w-full justify-between rounded-md px-3 py-2 text-sm font-semibold text-primaryText shadow-sm ring-1 ring-inset ring-secondaryText hover:bg-icons/20 ${
            isLoading ? "cursor-not-allowed" : "cursor-pointer"
          }`}
          id="menu-button"
          aria-expanded={isOpen}
          aria-haspopup="true"
          onClick={handleToggle}
          disabled={disabled || isLoading}
        >
          {value ? value : placeholder}
          <MdArrowDropDown className="-mr-1 h-5 w-5 text-secondaryText" />
        </button>
      </div>
      {isOpen && (
        <div
          className="absolute right-0 z-10 mt-2 w-56 origin-top-right divide-y divide-gray-100 rounded-md bg-white shadow-lg ring-1 ring-secondaryText ring-opacity-5 focus:outline-none"
          role="menu"
          aria-orientation="vertical"
          aria-labelledby="menu-button"
          tabIndex={-1}
        >
          <div className="py-1" role="none">
            <a
              href="#"
              className="text-gray-700 block px-4 py-2 text-sm"
              role="menuitem"
              tabIndex={-1}
              onClick={() => handleSelect("TF-IDF")}
            >
              TF-IDF
            </a>
            <a
              href="#"
              className="text-gray-700 block px-4 py-2 text-sm"
              role="menuitem"
              tabIndex={-1}
              onClick={() => handleSelect("BM25")}
            >
              BM25
            </a>
          </div>
          <div className="py-1" role="none">
            <a
              href="#"
              className="text-gray-700 block px-4 py-2 text-sm"
              role="menuitem"
              tabIndex={-1}
              onClick={() => handleSelect("TF-IDF+Letor")}
            >
              TF-IDF+Letor
            </a>
            <a
              href="#"
              className="text-gray-700 block px-4 py-2 text-sm"
              role="menuitem"
              tabIndex={-1}
              onClick={() => handleSelect("BM25+Letor")}
            >
              BM25+Letor
            </a>
          </div>
        </div>
      )}
    </div>
  );
};
