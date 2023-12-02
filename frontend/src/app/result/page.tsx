"use client";

import React from "react";
import Image from "next/image";
import { useState, useEffect } from "react";
import {
  Button,
  TextField,
  Dropdown,
  Container,
} from "../../components/elements";
import { Pagination } from "@mui/material";

export default function Result() {
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
    <main className="flex min-h-screen flex-col items-center justify-start gap-5 bg-primary">
      <div className="relative flex flex-col gap-4 items-center justify-center my-12 w-[90%]">
        <div className="flex gap-4 items-center w-full">
          <div className="w-min-[700px] flex flex-col items-end">
            <div
              className="w-fit text-4xl text-primaryText font-extrabold text-right"
              suppressHydrationWarning
            >
              {dateTime.toLocaleString("en-US", { timeStyle: "short" })}
            </div>
            <div className="w-fit text-xl text-secondaryText font-semibold text-right">
              {greeting}
            </div>
          </div>

          <div className="w-full flex flex-row bg-white py-3 px-5 items-center justify-center rounded-full">
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

        <div className="w-full">
          <svg
            width="100%"
            height="4"
            viewBox="0 0 1400 4"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <line x1="0" y1="0.5" x2="1400" y2="0.5" stroke="#F4F4E2" />
          </svg>
        </div>

        <div className="w-full flex text-sm items-center text-stone-400 ">
          <div>100 documents are retrieved in 0.0001 seconds.</div>
          <Image
            src="/CradrenFast.svg"
            alt="Cradren Fast"
            width={70}
            height={70}
          ></Image>
        </div>
        <div>
          <Container className="w-[70%]"></Container>
        </div>
        <div className="relative top-4">
          <Pagination count={10} shape="rounded" />
        </div>
      </div>
    </main>
  );
}
