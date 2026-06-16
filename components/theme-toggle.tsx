"use client"

import { Moon, Sun } from 'lucide-react'
import { Switch } from "@/components/ui/switch"
import { useTheme } from "@/components/theme-provider"

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-muted/30 hover:bg-muted/50 transition-all duration-200">
      <Switch
        checked={theme === "light"}
        onCheckedChange={toggleTheme}
        aria-label="Toggle theme"
      >
        {isDark ? (
          <Moon className={`h-3.5 w-3.5 text-black transform transition-all duration-200 ease-out ${isDark ? 'scale-110 opacity-100' : 'scale-95 opacity-90'}`} />
        ) : (
          <Sun className={`h-3.5 w-3.5 text-orange-500 transform transition-all duration-200 ease-out ${isDark ? 'scale-95 opacity-90' : 'scale-110 opacity-100'}`} />
        )}
      </Switch>
    </div>
  )
}
