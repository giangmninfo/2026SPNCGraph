'use client'

import type React from "react"
import { useRouter } from 'next/navigation'
import { useEffect, useState } from "react"
import { ThemeToggle } from "@/components/theme-toggle"

interface HeaderWithNavProps {
  title?: string
  showLogout?: boolean
  onLogout?: () => void
  rightContent?: React.ReactNode
}

export function HeaderWithNav({ title, showLogout = false, onLogout, rightContent }: HeaderWithNavProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const router = useRouter()

  useEffect(() => {
    const isLoggedIn = localStorage.getItem("isLoggedIn") === "true"
    setIsAuthenticated(isLoggedIn)
  }, [])

  const handleGNNClick = () => {
    if (isAuthenticated) {
      router.push("/classifier")
    } else {
      router.push("/")
    }
  }

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <button
          onClick={handleGNNClick}
          className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
        >
          <div className="px-3 py-2 rounded-full bg-[#B8DCFF] dark:bg-[#003D75] flex items-center justify-center">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary/70 rounded-full flex items-center justify-center" />
          </div>
          <div className="flex flex-col items-start">
            <span className="text-xs font-bold text-muted-foreground uppercase tracking-widest">GNN</span>
            {title && <h1 className="text-sm font-semibold leading-tight">{title}</h1>}
          </div>
        </button>

        <div className="flex items-center gap-4">
          {rightContent}
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
