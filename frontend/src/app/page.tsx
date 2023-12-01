"use client";

import React from "react";
import Image from "next/image";
import { useState, useEffect } from "react";
import { Button, TextField, Dropdown } from "../components/elements";

export default function Home() {
  const [dateTime, setDateTime] = useState(new Date());
  const [searchValue, setSearchValue] = useState("");
  const [methodValue, setMethodValue] = useState("");

  const hour = dateTime.getHours();
  let greeting;

  if (hour >= 5 && hour < 12) {
    greeting = "Good Morning";
  } else if (hour >= 12 && hour < 17) {
    greeting = "Good Afternoon";
  } else {
    greeting = "Good Evening";
  }

  useEffect(() => {
    const intervalId = setInterval(() => {
      setDateTime(new Date());
    }, 1000);

    return () => clearInterval(intervalId);
  }, []);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchValue(e.target.value);
  };

  const handleMethodChange = (selectedMethod: string) => {
    setMethodValue(selectedMethod);
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-5 p-24 bg-primary">
      <div className="flex flex-col gap-2 items-center">
        <div
          className="text-7xl text-primaryText font-black"
          suppressHydrationWarning
        >
          {dateTime.toLocaleString("en-US", { timeStyle: "short" })}
        </div>
        <div className="text-5xl text-secondaryText font-semibold">
          {greeting}
        </div>
      </div>

      <div className="flex flex-col w-full justify-center items-center">
        <Image
          src="/Cradren.svg"
          alt="Cradren"
          width={100}
          height={100}
        ></Image>

        <div className="relative top-[-42px] flex flex-row w-[60%] bg-white py-3 px-5 items-center justify-center rounded-full">
          <TextField
            className="w-full"
            placeholder="You are safe to pry here :)"
            value={searchValue}
            onChange={handleSearchChange}
          />

          <div className="flex flex-row gap-2 items-center justify-center">
            <Dropdown
              className="w-40"
              placeholder="Select Method"
              value={methodValue}
              onChange={handleMethodChange}
            ></Dropdown>
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
            <Button className="py-2 px-5">Search</Button>
          </div>
        </div>
      </div>
    </main>
  );
}
