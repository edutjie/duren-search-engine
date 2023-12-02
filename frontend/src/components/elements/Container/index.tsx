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
        className={`${className} px-6 py-4 bg-primaryContainer rounded-xl flex-col justify-center items-start gap-1 inline-flex ${
          isLoading
            ? "border-black disabled:bg-orange-dark"
            : "disabled:bg-cream-normal"
        } disabled:text-primaryContainer disabled:drop-shadow-none disabled:bg-transparent`}
      >
        <p className="text-primaryText text-base font-bold">
          Batman v Superman
        </p>
        <p className="line-clamp-3 text-stone-400 text-sm font-normal">
          Morbi vel nulla hendrerit, fringilla lorem et, volutpat nulla. Nam
          aliquam lorem non mauris tempor imperdiet. Duis fermentum sapien nisl,
          ac efficitur lacus commodo a. Aliquam a malesuada felis, non varius
          sem. Sed vel est ac nunc tincidunt mollis et a leo. Donec in ligula
          eget lorem euismod viverra eget eget nunc. Vivamus hendrerit ipsum
          nulla. Fusce tincidunt, justo a pretium elementum, nisl mi consectetur
          nibh, quis ullamcorper odio ante suscipit metus. Morbi eget arcu sit
          amet diam sollicitudin molestie ac vitae ligula. Donec feugiat et
          sapien eu feugiat. Vivamus posuere nulla sed bibendum volutpat. Ut
          elit ligula, consequat vel posuere sit amet, congue ac augue. Maecenas
          sagittis felis non neque laoreet sollicitudin. Donec quam magna,
          scelerisque quis sem ut, cursus sodales libero. In quis odio feugiat,
          molestie mi id, aliquet ipsum. Sed quis sem ipsum. ...
        </p>

        {isLoading ? (
          <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-inherit"></div>
        ) : (
          children
        )}
      </div>
    </>
  );
};
