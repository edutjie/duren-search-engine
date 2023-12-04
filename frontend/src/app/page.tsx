"use client";

import axios from "axios";
import Image from "next/image";
import { useRouter, useSearchParams } from "next/navigation";
import React, { useEffect, useState } from "react";
import { Button, Container, Dropdown, TextField } from "../components/elements";
import { HistoryProps } from "./interfaces";

export default function Home() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [dateTime, setDateTime] = useState(new Date());
  const [searchValue, setSearchValue] = useState("");
  const [methodValue, setMethodValue] = useState("");
  const [historyData, setHistoryData] = useState<Array<HistoryProps>>([]);

  useEffect(() => {
    const deviceId = localStorage.getItem("device_id");

    if (!deviceId) {
      const newDeviceId = Math.random().toString(36).substring(2, 15);
      localStorage.setItem("device_id", newDeviceId);
    }
  }, []);

  const hour = dateTime.getHours();
  let greeting: string;

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

  const handleSearchClick = async () => {
    try {
      const sanitizedMethod = methodValue.replace(/-/g, "");
      const lowercaseMethod = sanitizedMethod.toLowerCase();

      const params = new URLSearchParams(searchParams);
      params.delete("query");
      params.delete("method");
      params.set("query", searchValue);
      params.set("method", !!lowercaseMethod ? lowercaseMethod : "bm25");

      router.push("/result" + "?" + params.toString());
    } catch (error) {
      // Handle errors
      console.error("Search error:", error);
    }
  };

  const handleHistory = async () => {
    try {
      const deviceId = localStorage.getItem("device_id");
      const response = await axios.get(
        `${process.env.NEXT_PUBLIC_API_URL}/history`,
        {
          params: {
            device_id: deviceId,
          },
        }
      );
      setHistoryData(response.data.data);
      console.log("History success:", response.data);
    } catch (error) {
      console.error("History error:", error);
    }
  };

  useEffect(() => {
    handleHistory();
  }, []);

  const formatTime = (time: string) => {
    const timeStr = time.split(" ")[1];
    const timeOnly = timeStr.replaceAll("-", ":");
    return timeOnly;
  };

  return (
    <main className="flex min-h-screen items-center justify-center gap-8 p-24 bg-primary">
      <div className="w-full flex flex-col items-center justify-center gap-5">
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

          <div className="relative top-[-42px] flex flex-row w-full bg-white py-3 px-5 items-center justify-center rounded-full">
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
              <Button
                className="bg-primaryText py-2 px-5"
                onClick={handleSearchClick}
              >
                Search
              </Button>
            </div>
          </div>
        </div>
      </div>
      <div className="w-[25%] h-[500px] overflow-y-auto rounded-2xl">
        <div className="flex flex-col w-full justify-center items-center gap-4">
          {historyData.map(data => (
            <Container className="flex-col w-full gap-2" key={data.date}>
              <div className="text-primaryText text-lg font-bold">
                {data.date}
              </div>
              {data.queries.map((item, index) => (
                <div className="flex w-full" key={index}>
                  <div className="flex w-full justify-between gap-4">
                    <div className="text-stone-500 text-base font-semibold">
                      {item.query}
                    </div>
                    <div className="text-stone-400 text-sm font-normal">
                      {formatTime(item.time)}
                    </div>
                  </div>
                </div>
              ))}
            </Container>
          ))}
        </div>
      </div>
    </main>
  );
}
