using Documenter
using RankRevealing

# Set the right metadata for doctests
DocMeta.setdocmeta!(RankRevealing, :DocTestSetup, :(using RankRevealing); recursive=true)

makedocs(
  sitename = "RankRevealing.jl",
  format   = Documenter.HTML(
    assets   = ["assets/favicon.ico"],
  ),
  modules  = [RankRevealing]
)

# Deplou site to Github Pages
if !("local" in ARGS)
  deploydocs(
    repo = "github.com/iagoleal/RankRevealing.jl.git",
    devurl = "latest",
  )
end
