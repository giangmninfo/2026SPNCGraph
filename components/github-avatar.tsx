"use client"

import { useMemo } from "react"

interface GitHubAvatarProps {
  username?: string
  size?: number
}

export const getAvatarColorForUser = (username: string): string => {
  if (!username || typeof username !== "string") return "#4ECDC4"
  
  const colors = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
    "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52C7B8",
    "#FF8C42", "#A8E6CF", "#FFD93D", "#6BCB77", "#4D96FF",
    "#FF6348", "#20C997", "#FFB627", "#FF6584", "#9B59B6",
  ]
  let hash = 0
  for (let i = 0; i < username.length; i++) {
    hash = username.charCodeAt(i) + ((hash << 5) - hash)
  }
  return colors[Math.abs(hash) % colors.length]
}

export function GitHubAvatar({ username = "", size = 120 }: GitHubAvatarProps) {
  const { blockColor, blocks } = useMemo(() => {
    const selectedColor = getAvatarColorForUser(username || "guest")

    const grid = Array(11 * 11).fill(0)
    
    grid[2 * 11 + 2] = 1; grid[3 * 11 + 2] = 1;
    grid[2 * 11 + 8] = 1; grid[3 * 11 + 8] = 1; 
    grid[2 * 11 + 5] = 1; 
    grid[3 * 11 + 5] = 1; 
    grid[4 * 11 + 5] = 1; 
    grid[5 * 11 + 5] = 1; 
    grid[6 * 11 + 5] = 1; 
    grid[6 * 11 + 4] = 1;

    grid[8 * 11 + 3] = 1; 
    grid[8 * 11 + 7] = 1; 
    grid[9 * 11 + 4] = 1; grid[9 * 11 + 5] = 1; grid[9 * 11 + 6] = 1;

    return { blockColor: selectedColor, blocks: grid }
  }, [username])

  return (
    <div
      className="flex items-center justify-center border border-gray-200"
      style={{
        width: size,
        height: size,
        borderRadius: "15%",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(11, 1fr)",
          width: "80%", 
          height: "80%",
        }}
      >
        {blocks.map((isActive, idx) => (
          <div
            key={idx}
            style={{
              backgroundColor: isActive ? blockColor : "transparent",
              borderRadius: "1px", 
            }}
          />
        ))}
      </div>
    </div>
  )
}