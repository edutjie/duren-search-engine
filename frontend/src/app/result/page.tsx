"use client";

import { Pagination } from "@mui/material";
import axios from "axios";
import Image from "next/image";
import { useRouter, useSearchParams } from "next/navigation";
import React, { useEffect, useState } from "react";
import {
  Button,
  Container,
  Dropdown,
  TextField,
} from "../../components/elements";
import { SearchResultProps } from "./interfaces";

export default function Result() {
  const [dateTime, setDateTime] = useState<Date>(new Date());
  const [searchValue, setSearchValue] = useState<string>("");
  const [methodValue, setMethodValue] = useState<string>("");
  const [searchResult, setSearchResult] = useState<Array<SearchResultProps>>(
    []
  );

  const router = useRouter();
  const searchParams = useSearchParams();
  let pageGiven = "1";

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

  const handleSearchClick = () => {
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
      console.error("Search error:", error);
    }
  };

  const handleDocumentClick = (doc_id: string) => {
    try {
      const params = new URLSearchParams(searchParams);
      params.delete("doc_id");
      params.set("doc_id", doc_id);
      router.push("/details" + "?" + params.toString());
      // console.log(doc_id);
    } catch (error) {
      console.error("Search error:", error);
    }
  };

  const handleSearchResult = async () => {
    const methodGiven = searchParams.get("method");
    const queryGiven = searchParams.get("query");
    try {
      const deviceId = localStorage.getItem("device_id");
      console.log(deviceId)
      if (methodGiven?.includes("+")) {
        const substring = methodGiven.split("+");
        const method = substring[0];
        const response = await axios.get(
          `${process.env.NEXT_PUBLIC_API_URL}/search/${method}`,
          {
            params: {
              query: queryGiven,
              is_letor: true,
              page: pageGiven,
              device_id: deviceId,
            },
          }
        );
        setSearchResult(response.data.data);
        console.log("Search success:", response.data);
      } else {
        const response = await axios.get(
          `${process.env.NEXT_PUBLIC_API_URL}/search/${methodGiven}`,
          {
            params: {
              query: queryGiven,
              is_letor: false,
              page: pageGiven,
              device_id: deviceId,
            },
          }
        );
        setSearchResult(response.data.data);
        console.log("Search success:", response.data);
      }
    } catch (error) {
      // Handle errors
      console.error("Search error:", error);
    }
  };

  useEffect(() => {
    handleSearchResult();
  }, [searchParams]);

  const handlePaginationChange = (
    event: React.ChangeEvent<unknown>,
    value: number
  ) => {
    pageGiven = value.toString();
    handleSearchResult();
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-start gap-5 bg-primary">
      <div className="relative flex flex-col gap-4 items-center justify-center my-12 w-[90%]">
        <div className="flex gap-4 items-center w-full">
          <div className="w-[15%] flex flex-col items-end">
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
              <Button
                className="bg-primaryText py-2 px-5"
                onClick={handleSearchClick}
              >
                Search
              </Button>
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

        <div className="w-full flex text-sm items-center justify-center text-stone-400 ">
          <div>100 documents are retrieved in 0.0001 seconds.</div>
          <Image
            src="/CradrenFast.svg"
            alt="Cradren Fast"
            width={70}
            height={70}
            onClick={() => {
              router.push("/");
            }}
          ></Image>
        </div>
        <div className="flex flex-col w-full justify-center items-center gap-4">
          {searchResult.map((result, index) => (
            <div
              className="flex w-[70%] gap-4"
              key={index}
              onClick={() => handleDocumentClick(result.id)}
            >
              <Container className="flex w-full">
                <div className="text-primaryText text-base font-bold">
                  {result.title}
                </div>
                <div className="line-clamp-3 text-stone-400 text-sm font-normal">
                  {result.preview}
                </div>
              </Container>
            </div>
          ))}
        </div>
        <div className="relative top-4">
          <Pagination
            count={10}
            shape="rounded"
            onChange={handlePaginationChange}
          />
        </div>
      </div>
    </main>
  );
}
