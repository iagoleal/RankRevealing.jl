using Documenter
using RankRevealing

# Set the right metadata for doctests
DocMeta.setdocmeta!(RankRevealing, :DocTestSetup, :(using RankRevealing); recursive=true)

makedocs(
    sitename = "RankRevealing.jl",
    format = Documenter.HTML(),
    modules = [RankRevealing]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
