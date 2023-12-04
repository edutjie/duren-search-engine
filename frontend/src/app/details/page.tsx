"use client";

import React, { useCallback } from "react";
import Image from "next/image";
import { useState, useEffect } from "react";
import { Button, Container } from "../../components/elements";
import axios from "axios";
import { useRouter, useSearchParams } from "next/navigation";

export default function Details() {
  const [documentDetail, setDocumentDetail] = useState([]);
  const [relDocsList, setRelDocsList] = useState([]);

  const router = useRouter();
  const searchParams = useSearchParams();
  const doc_idGiven = searchParams.get("doc_id");

  const handleBack = () => {
    router.back();
  };

  const handleHome = () => {
    router.push("/");
    searchParams.delete();
  };

  const handleDocument = async () => {
    if (doc_idGiven) {
      try {
        const response = await axios.get(
          `${process.env.NEXT_PUBLIC_API_URL}/document/${doc_idGiven}`
        );
        setDocumentDetail(response.data);
        console.log("Search success:", response.data);
      } catch (error) {
        // Handle errors
        console.error("Search error:", error);
      }
    } else {
      console.log("Search error: No document id given");
    }
  };

  const handleRelDocs = async () => {
    if (doc_idGiven) {
      try {
        const response = await axios.get(
          `${process.env.NEXT_PUBLIC_API_URL}/related/${doc_idGiven}`
        );
        setRelDocsList(response.data.data);
        console.log("Search success:", response.data);
      } catch (error) {
        // Handle errors
        console.error("Search error:", error);
      }
    } else {
      console.log("Search error: No document id given");
    }
  };

  useEffect(() => {
    handleDocument();
  }, []);

  useEffect(() => {
    handleRelDocs();
  }, []);

  const createQueryString = useCallback(
    (name: string, value: string) => {
      const params = new URLSearchParams(searchParams);
      params.set(name, value);

      return params.toString();
    },
    [searchParams]
  );

  const handleDocumentClick = (doc_id: string) => {
    try {
      searchParams.delete();
      router.push("/details" + "?" + createQueryString("doc_id", doc_id));
      // console.log(doc_id);
    } catch (error) {
      console.error("Search error:", error);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-start gap-5 bg-primary">
      <div className="relative flex flex-col gap-8 items-center justify-center my-12 w-[70%]">
        <div className="w-full flex flex-row bg-white py-3 px-5 items-center justify-center rounded-full">
          <Button className="py-2 px-5 bg-primaryText" onClick={handleBack}>
            Back
          </Button>
          <div className="text-primaryText text-2xl font-bold w-full text-center">
            {documentDetail.title}
          </div>
          <Button
            className="py-2 px-5 bg-primaryContainer text-primaryText"
            onClick={handleHome}
          >
            Home
          </Button>
        </div>
        <div className="w-full text-stone-500 text-lg px-12">
          {documentDetail.content}
        </div>
        <div className="flex flex-col w-full px-12 justify-center items-center gap-5">
          <Image
            src="/Cradren.svg"
            alt="Cradren"
            width={60}
            height={60}
          ></Image>
          <div className="text-secondaryText text-xl font-bold">
            Relevant Documents:
          </div>
          {relDocsList.map((result, index) => (
            <div
              className="flex w-full gap-4"
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
      </div>
    </main>
  );
}
